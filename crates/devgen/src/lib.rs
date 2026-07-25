//! `gemma4` — compile the REAL Gemma-4 31B (or 12B) prefill network into a device packet
//! program, straight from the HuggingFace checkpoint.
//!
//! # DEPRECATED
//!
//! Superseded by `plowc --hf-dir <dir>`, which compiles the same checkpoints
//! through the shared driver pipeline. Only built with
//! `--features legacy-gemma-bins`; slated for removal.
//!
//! Reads `config.json` and the safetensors index, and emits packets whose weight tensors
//! are named EXACTLY as the checkpoint names them, so the runtime can bind them by name
//! and hard-fail on anything missing. A silently-absent weight is the worst failure mode
//! in this whole stack: the model still produces fluent text, just wrong text.
//!
//! # The spec, verified against the checkpoint and modeling_gemma4.py — not from memory
//!
//! Every one of these is a silent fluent-but-wrong bug if you get it wrong:
//!
//! * **RMSNorm has NO `+1`.** `x * pow(mean(x^2) + eps, -0.5) * w`, eps INSIDE the power.
//!   Gemma 1/2/3 used `(1 + w)` with zero-init weights; Gemma 4 is ones-init and dropped it.
//! * **Attention scale is 1.0.** There is no `1/sqrt(head_dim)` anywhere — the trained
//!   `q_norm` absorbs it (`self.scaling = 1.0` in the reference).
//! * **`v_norm` is a WEIGHTLESS RMSNorm** over head_dim, applied to V on every layer, and
//!   it has no checkpoint tensor — so it is the easiest thing in the model to omit.
//! * **Full-attention layers have NO `v_proj`** (`attention_k_eq_v: true`). V comes from the
//!   RAW k_proj output: `K = RoPE(k_norm(kv))`, `V = v_norm(kv)`, both from one projection.
//!   Confirmed in the checkpoint: layer 5 ships q/k/o_proj and no v_proj.
//! * **Full layers use `global_head_dim` = 512 and `num_global_key_value_heads` = 4**, not
//!   the sliding layers' 256/16.
//! * **Partial RoPE on full layers**: `rope_angles = int(0.25 * 512 // 2) = 64`, so
//!   `inv_freq[i] = 1e6^(-2i/512)` for i < 64 and **ZERO for i in [64, 256)** — those dims
//!   pass through unrotated (NoPE). Rotated pairs are `(i, i+256)`, not `(i, i+64)`.
//! * **MLP is GeGLU** (gelu_pytorch_tanh), not SwiGLU.
//! * **Sandwich norms**: the residual is added AFTER the post-norm.
//! * **`layer_scalar`** is a learned `[1]` tensor multiplying the whole hidden state at the
//!   end of each layer. We fold it into the second residual's scale — algebraically the
//!   same thing — which means the COMPILER has to read it out of the checkpoint.
//! * **Embedding scale is the BF16-ROUNDED sqrt(hidden)**: 73.5, not 73.3212.
//! * **Tied lm_head**, then `logits = 30 * tanh(logits / 30)`.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use costmodel::hwspec;
use packet::dev::{DevInst, DevOp, TENSOR_NONE};
use packet::devbuild::{Builder, Dep, Model};
use packet::rope::{GenTensor, RopeScale};
use serde_json::Value;

/// Flash-decode GQA fusion factor on FULL-attention layers. **This must equal the
/// kernel constant** (`PLOW_NV_FA_GF` on sm_120, `PLOW_FA_GF_FULL` on AMD) or the
/// compiler and kernel disagree about how many query heads one work item carries.
/// Previously a bare `let gf = 2` used only in an assertion — it never reached the
/// packet, so it read like a knob and controlled nothing. Qwen3 is gqa=4 and the
/// sm_120 build ships GF=4 (worth a measured 1.71x on flash-decode); GF=2 is a
/// Gemma artifact. The binding invariant is `gqa_local % FA_GF_FULL == 0`.
const FA_GF_FULL: u32 = 2;

/// Greatest common divisor (Euclid). Used to grid-align the full-layer flash-decode
/// nsplit to the resident-block count (T9b).
fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a.max(1)
}

const BF16: u64 = 2;
const F32: u64 = 4;
const I32: u64 = 4;
const BM: u32 = 256;
const BN: u32 = 256;

/// Which checkpoint architecture we are compiling. The three differ in tensor NAMING, norm
/// TOPOLOGY, activation, attention geometry, and RoPE — see the module docstring for Gemma and
/// `plans/multi-model-baseline.md` for the Llama/Qwen deltas. The Gemma-4 path is unchanged.
#[derive(Clone, Copy, PartialEq)]
enum Arch {
    Gemma4,
    Llama,
    Qwen3,
}


struct Cfg {
    arch: Arch,
    hidden: u32,
    inter: u32,
    layers: u32,
    heads: u32,
    hd_slide: u32,
    hd_full: u32,
    kvh_slide: u32,
    kvh_full: u32,
    window: u32,
    eps: f32,
    vocab: u32,
    softcap: f32,
    is_full: Vec<bool>,
    theta_slide: f64,
    theta_full: f64,
    rope_frac_full: f64,
    rope_scale: RopeScale,
    // Arch switches (Gemma values preserve the old behaviour exactly).
    attn_scale: f32,   // Gemma 1.0 (q_norm absorbs it); Llama/Qwen 1/sqrt(head_dim)
    emb_scale: f32,    // Gemma bf16_round(sqrt(hidden)); Llama/Qwen 1.0
    mlp_act: u32,      // 0 = gelu_tanh (Gemma), 1 = silu (Llama/Qwen)
    has_qk_norm: bool, // Gemma & Qwen true; Llama false
    has_v_norm: bool,  // Gemma weightless v_norm; Llama/Qwen false
    k_eq_v: bool,      // Gemma full layers share k_proj as V; Llama/Qwen false
    tied: bool,        // reuse embed_tokens as lm_head (Gemma, Qwen); Llama has lm_head.weight
    prefix: String,    // weight-name prefix: "model.language_model." or "model."
    // Tensor-parallel degree (Megatron sharding). 1 = single-GPU (current path, byte-identical).
    // >1 emits a DECODE-ONLY sharded blob (plans/tp-design.md §3): column-parallel q/k/v/gate/up/
    // lm_head, row-parallel o_proj/down with an XReduce all-reduce after each, attention split by
    // heads. All ranks run the ONE blob; tp-host binds each rank's 1/N weight slice and sets
    // PlowProgram.rank/n_gpu/peer_scratch/xctr. Set from --tp in main() after cfg_from.
    tp: u32,
    // Gemma-4 26B-A4B sparse-MoE (`enable_moe_block`). Every layer is a HYBRID dense+MoE block:
    // the dense MLP (inter) AND the top-`top_k`-of-`n_exp` softmax-routed experts (moe_inter),
    // summed via the h1+h2 sandwich (plans/rtx-08-gemma4-moe-26b.md). Decode-only for now.
    moe: bool,
    n_exp: u32,     // 128 routed experts
    top_k: u32,     // 8 experts/token
    moe_inter: u32, // 704 per-expert intermediate
}

fn cfg_from(dir: &Path) -> Cfg {
    let v: Value =
        serde_json::from_slice(&std::fs::read(dir.join("config.json")).expect("config.json"))
            .unwrap();
    // Gemma-4 multimodal nests everything under `text_config` (prefix
    // "model.language_model."); the text-only "-it-text" re-export is FLAT with
    // model_type "gemma4_text" (prefix "model."). Same weights, two namings.
    if v.get("text_config").is_some() {
        return cfg_gemma(&v, false);
    }
    let mt = v["model_type"].as_str().unwrap_or("");
    if mt == "gemma4_text" {
        return cfg_gemma(&v, true);
    }
    let arch = match mt {
        "qwen3" => Arch::Qwen3,
        "llama" => Arch::Llama,
        other => panic!("unsupported model_type {other:?}"),
    };
    cfg_llama_qwen(&v, arch)
}

/// The original Gemma-4 config parse, verbatim — do not regress it. `flat` selects
/// the text-only re-export (fields at the root, "model." prefix) vs the multimodal
/// checkpoint (fields under `text_config`, "model.language_model." prefix).
fn cfg_gemma(v: &Value, flat: bool) -> Cfg {
    let t = if flat { v } else { &v["text_config"] };
    let g = |k: &str| t[k].as_u64().unwrap() as u32;
    let lt: Vec<bool> = t["layer_types"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_str().unwrap() == "full_attention")
        .collect();
    let rp = &t["rope_parameters"];
    Cfg {
        arch: Arch::Gemma4,
        hidden: g("hidden_size"),
        inter: g("intermediate_size"),
        layers: g("num_hidden_layers"),
        heads: g("num_attention_heads"),
        hd_slide: g("head_dim"),
        hd_full: g("global_head_dim"),
        kvh_slide: g("num_key_value_heads"),
        // Nullable: the E-series ships `num_global_key_value_heads: null` and reuses the
        // sliding count. `g()` would panic on the None; fall back the way
        // hf_config::synth_gemma already does, so an unsupported checkpoint reaches the
        // coverage gate and gets a diagnosis instead of an `Option::unwrap` backtrace.
        kvh_full: t["num_global_key_value_heads"]
            .as_u64()
            .map(|x| x as u32)
            .unwrap_or_else(|| g("num_key_value_heads")),
        window: g("sliding_window"),
        eps: t["rms_norm_eps"].as_f64().unwrap() as f32,
        vocab: g("vocab_size"),
        softcap: t["final_logit_softcapping"].as_f64().unwrap() as f32,
        theta_slide: rp["sliding_attention"]["rope_theta"].as_f64().unwrap(),
        theta_full: rp["full_attention"]["rope_theta"].as_f64().unwrap(),
        rope_frac_full: rp["full_attention"]["partial_rotary_factor"]
            .as_f64()
            .unwrap(),
        is_full: lt,
        rope_scale: RopeScale::None,
        attn_scale: 1.0,
        emb_scale: bf16_round((g("hidden_size") as f32).sqrt()),
        mlp_act: 0,
        has_qk_norm: true,
        has_v_norm: true,
        k_eq_v: true,
        tied: true,
        prefix: if flat {
            "model."
        } else {
            "model.language_model."
        }
        .to_string(),
        tp: 1,
        // 26B-A4B: enable_moe_block=true, num_experts=128, top_k_experts=8, moe_inter=704.
        // 12B/31B: field absent -> dense-only (moe=false).
        moe: t["enable_moe_block"].as_bool().unwrap_or(false),
        n_exp: t["num_experts"].as_u64().unwrap_or(0) as u32,
        top_k: t["top_k_experts"].as_u64().unwrap_or(0) as u32,
        moe_inter: t["moe_intermediate_size"].as_u64().unwrap_or(0) as u32,
    }
}

/// Llama-3.1 / Qwen3: flat config, all-global attention, simple pre-norm, SwiGLU.
fn cfg_llama_qwen(v: &Value, arch: Arch) -> Cfg {
    let g = |k: &str| v[k].as_u64().unwrap() as u32;
    let hidden = g("hidden_size");
    let heads = g("num_attention_heads");
    // Qwen carries head_dim explicitly (and it is NOT hidden/heads: 2560/32 != 128); Llama omits
    // it, so it is hidden/heads = 128.
    let hd = v["head_dim"]
        .as_u64()
        .map(|x| x as u32)
        .unwrap_or(hidden / heads);
    let layers = g("num_hidden_layers");
    let theta = v["rope_theta"].as_f64().unwrap();
    // llama3 rope scaling (Llama-3.1); Qwen has rope_scaling: null.
    let rope_scale = match v.get("rope_scaling").and_then(|r| r.as_object()) {
        Some(r) if r.get("rope_type").and_then(|x| x.as_str()) == Some("llama3") => {
            RopeScale::Llama3 {
                factor: r["factor"].as_f64().unwrap(),
                low: r["low_freq_factor"].as_f64().unwrap(),
                high: r["high_freq_factor"].as_f64().unwrap(),
                orig: r["original_max_position_embeddings"].as_f64().unwrap(),
            }
        }
        // A rope_type we do not implement must be a HARD FAILURE. Falling through
        // to RopeScale::None here compiles silently-wrong rope tables, which
        // produce fluent-but-wrong text with no crash and no numeric gate that
        // catches it. Both gemma-4-12B and gemma-4-31B hit this arm.
        Some(r) => {
            let ty = r
                .get("rope_type")
                .and_then(|x| x.as_str())
                .unwrap_or("<missing>");
            panic!(
                "unsupported rope_type {ty:?} in rope_scaling: this compiler implements \
                 only \"llama3\". Compiling it as unscaled would emit wrong rope tables \
                 and produce fluent-but-wrong output. Add a RopeScale arm for {ty:?}."
            );
        }
        None => RopeScale::None,
    };
    Cfg {
        arch,
        hidden,
        inter: g("intermediate_size"),
        layers,
        heads,
        hd_slide: hd,
        hd_full: hd,
        kvh_slide: g("num_key_value_heads"),
        kvh_full: g("num_key_value_heads"),
        window: 0, // all-global: no sliding window
        eps: v["rms_norm_eps"].as_f64().unwrap() as f32,
        vocab: g("vocab_size"),
        softcap: 0.0, // no final-logit softcapping
        is_full: vec![true; layers as usize],
        theta_slide: theta,
        theta_full: theta,
        rope_frac_full: 1.0, // full rotary
        rope_scale,
        attn_scale: 1.0 / (hd as f32).sqrt(),
        emb_scale: 1.0, // no embedding scaling
        mlp_act: 1,     // SwiGLU (silu)
        has_qk_norm: arch == Arch::Qwen3,
        has_v_norm: false,
        k_eq_v: false,
        tied: v["tie_word_embeddings"].as_bool().unwrap_or(false),
        prefix: "model.".to_string(),
        tp: 1,
        moe: false, // Llama/Qwen3 dense here
        n_exp: 0,
        top_k: 0,
        moe_inter: 0,
    }
}

/// Read the `layer_scalar` values out of the checkpoint.
///
/// The RESIDUAL op takes its scale as an IMMEDIATE in the packet, not as a tensor — so the
/// compiler, not the runtime, has to know it. That means plowc reads the safetensors
/// headers. This is the right place for it: `layer_scalar` is a compile-time constant of
/// the network, exactly like the tile size.
/// Discover the checkpoint's shard files, newest-bug-first.
///
/// FIXED: this used to read `model.safetensors.index.json` unconditionally and
/// panic on the Gemma-4 12B checkpoint, which is a single unsharded
/// `model.safetensors` with **no index file at all**. It also trusted the index
/// on the 31B *partial* checkpoint, whose `model.safetensors.index.json` names
/// files (`model-0000N-of-00002.safetensors`) that do not exist on disk — only
/// the `.partial.safetensors` ones do.
///
/// So don't trust the index: enumerate what is actually there. Same resolution
/// order as `plowrt::memory::container::Safetensors::open_dir` — a complete
/// non-partial shard set, else a partial set, else single-file
/// `model.safetensors`.
fn shard_files(dir: &Path) -> Vec<PathBuf> {
    let mut sets: HashMap<(u32, bool), Vec<(u32, PathBuf)>> = HashMap::new();
    let mut single = None;
    for ent in std::fs::read_dir(dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display())) {
        let ent = ent.unwrap();
        let fname = ent.file_name();
        let Some(f) = fname.to_str() else { continue };
        if f == "model.safetensors" {
            single = Some(ent.path());
            continue;
        }
        // Suffix must be exactly ".safetensors" so sidecars like
        // "model-00001-of-00002.safetensors.header.json" don't match.
        let Some(rest) = f.strip_prefix("model-") else {
            continue;
        };
        let Some((i, rest)) = rest.split_once("-of-") else {
            continue;
        };
        let (t, partial) = match rest.strip_suffix(".partial.safetensors") {
            Some(t) => (t, true),
            None => match rest.strip_suffix(".safetensors") {
                Some(t) => (t, false),
                None => continue,
            },
        };
        let (Ok(i), Ok(t)) = (i.parse::<u32>(), t.parse::<u32>()) else {
            continue;
        };
        sets.entry((t, partial)).or_default().push((i, ent.path()));
    }
    let mut complete: Vec<_> = sets
        .iter()
        .filter(|((t, _), v)| v.len() as u32 == *t)
        .collect();
    // Prefer the non-partial set when both are complete at the same total.
    complete.sort_by_key(|((t, p), _)| (*p, *t));
    if complete.len() > 1 {
        let keys: Vec<_> = complete.iter().map(|(k, _)| **k).collect();
        assert!(
            keys.len() == 2 && keys[0].0 == keys[1].0 && keys[0].1 != keys[1].1,
            "{}: ambiguous checkpoint — {} complete shard sets {keys:?}; a stray shard-named \
             file silently changes what loads",
            dir.display(),
            keys.len()
        );
    }
    if let Some((_, v)) = complete.first() {
        let mut v = (*v).clone();
        v.sort_by_key(|(i, _)| *i);
        return v.into_iter().map(|(_, p)| p).collect();
    }
    if let Some((k, v)) = sets.iter().next() {
        panic!(
            "{}: incomplete shard set (-of-{:05}{}): {} of {} present",
            dir.display(),
            k.0,
            if k.1 { " .partial" } else { "" },
            v.len(),
            k.0
        );
    }
    // THE fallback that was missing: single-file, no index (Gemma-4 12B).
    vec![single.unwrap_or_else(|| {
        panic!(
            "{}: no safetensors checkpoint (looked for \
             model-{{i}}-of-{{n}}[.partial].safetensors and model.safetensors)",
            dir.display()
        )
    })]
}

fn layer_scalars(dir: &Path, layers: u32, prefix: &str) -> Vec<f32> {
    // name -> shard file, built from the files that actually exist rather than
    // from an index that may be absent (12B) or stale (31B partial).
    let mut hdr_cache: HashMap<PathBuf, (Value, u64)> = HashMap::new();
    let mut map: HashMap<String, PathBuf> = HashMap::new();
    for p in shard_files(dir) {
        // Read ONLY the header. The old code did `fs::read(&p)` — the whole
        // shard — to fetch a handful of `[1]` scalars; that is 23 GB of I/O on
        // the 12B checkpoint.
        use std::io::Read;
        let mut f = std::fs::File::open(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
        let mut len = [0u8; 8];
        f.read_exact(&mut len)
            .unwrap_or_else(|e| panic!("{}: {e}", p.display()));
        let n = u64::from_le_bytes(len);
        let mut hbuf = vec![0u8; n as usize];
        f.read_exact(&mut hbuf)
            .unwrap_or_else(|e| panic!("{}: {e}", p.display()));
        let h: Value = serde_json::from_slice(&hbuf)
            .unwrap_or_else(|e| panic!("{}: bad safetensors header: {e}", p.display()));
        for k in h.as_object().expect("header object").keys() {
            if k != "__metadata__" {
                map.insert(k.clone(), p.clone());
            }
        }
        hdr_cache.insert(p, (h, 8 + n));
    }
    let mut out = Vec::with_capacity(layers as usize);
    for l in 0..layers {
        let name = format!("{prefix}layers.{l}.layer_scalar");
        let path = map
            .get(&name)
            .unwrap_or_else(|| panic!("checkpoint has no {name}"))
            .clone();
        let (hdr, data0) = &hdr_cache[&path];
        let (data0, path) = (*data0, path.clone());
        let ent = &hdr[&name];
        assert_eq!(ent["dtype"].as_str().unwrap(), "BF16", "{name} dtype");
        let off = ent["data_offsets"][0].as_u64().unwrap();
        let mut f = std::fs::File::open(path).unwrap();
        use std::io::{Read, Seek, SeekFrom};
        f.seek(SeekFrom::Start(data0 + off)).unwrap();
        let mut b = [0u8; 2];
        f.read_exact(&mut b).unwrap();
        let bits = (u16::from_le_bytes(b) as u32) << 16;
        out.push(f32::from_bits(bits));
    }
    out
}

/// Every `prefix*` tensor name the checkpoint actually ships.
fn ckpt_names(dir: &Path, prefix: &str) -> HashSet<String> {
    let mut out = HashSet::new();
    for p in shard_files(dir) {
        use std::io::Read;
        let mut f = std::fs::File::open(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
        let mut len = [0u8; 8];
        f.read_exact(&mut len)
            .unwrap_or_else(|e| panic!("{}: {e}", p.display()));
        let mut hbuf = vec![0u8; u64::from_le_bytes(len) as usize];
        f.read_exact(&mut hbuf)
            .unwrap_or_else(|e| panic!("{}: {e}", p.display()));
        let h: Value = serde_json::from_slice(&hbuf)
            .unwrap_or_else(|e| panic!("{}: bad safetensors header: {e}", p.display()));
        for k in h.as_object().expect("header object").keys() {
            if k != "__metadata__" && k.starts_with(prefix) {
                out.insert(k.clone());
            }
        }
    }
    out
}

/// Bidirectional checkpoint coverage gate.
///
/// `--hf-dir` has had this since day one (`hf_config::validate_against_checkpoint`,
/// and `tests/hf_dir_compile.rs` calls the two failure modes out by name), but THIS
/// binary — the one every asset-build script actually runs — had no check at all. It
/// declares weights by name and never reads the checkpoint back, and the runtime's
/// only net (`plowrt::memory::container`) errors on a MISSING name, never on an
/// unused one. That is a pull model: extra checkpoint tensors are simply never
/// looked up.
///
/// Measured consequence, gemma-4-E4B-it: after one null-default the emitter produced
/// a clean, loadable, warning-free packet that had silently dropped **5.4 GiB** of
/// per-layer-embedding weights (reporting `weights 8.6 GiB` for a 14.0 GiB model).
/// It would have loaded and generated fluent, wrong text. Both directions matter:
/// the forward check catches a typo'd/renamed weight, the reverse check catches an
/// architecture the emitter does not implement.
///
/// Only `prefix*` names participate. Activations (`act.`), inputs (`in.`), KV rings
/// (`kv.`), compiler-materialised tables (rope) and the fp8 twins (`fp8/`, which live
/// in a sibling directory, not `dir`) are all out of scope by construction.
fn validate_coverage(dir: &Path, prefix: &str, declared: &[String]) -> Result<(), String> {
    let ckpt = ckpt_names(dir, prefix);
    // A weight is covered if the plan binds EITHER the bf16 tensor or its fp8 twin.
    // Under PLOW_FP8 the projections are declared as `fp8/<name>` (the twins live in a
    // sibling dir, so they are not in `ckpt`) and the bf16 original is deliberately
    // superseded — counting it "uncovered" would fail every fp8 build.
    let want: HashSet<&str> = declared
        .iter()
        .map(|s| s.strip_prefix("fp8/").unwrap_or(s.as_str()))
        .filter(|n| n.starts_with(prefix))
        .collect();

    // Forward: only bf16 names are resolvable against `dir`. fp8 twins are checked by
    // the loader against the twin directory, not here.
    let mut missing: Vec<&str> = declared
        .iter()
        .map(|s| s.as_str())
        .filter(|n| n.starts_with(prefix) && !ckpt.contains(*n))
        .collect();
    // `layer_scalar` is read by `layer_scalars()` as a compile-time immediate and
    // folded into the residual epilogue, so it is legitimately never declared.
    let mut uncovered: Vec<&str> = ckpt
        .iter()
        .map(|s| s.as_str())
        .filter(|n| !want.contains(*n) && !n.ends_with(".layer_scalar"))
        .collect();
    if missing.is_empty() && uncovered.is_empty() {
        return Ok(());
    }
    missing.sort_unstable();
    uncovered.sort_unstable();

    let sample = |v: &[&str]| {
        v.iter()
            .take(8)
            .copied()
            .collect::<Vec<_>>()
            .join("\n    ")
    };
    let mut e = String::from("checkpoint coverage failed:\n");
    if !missing.is_empty() {
        e.push_str(&format!(
            "  plan weights NOT in the checkpoint ({}):\n    {}\n",
            missing.len(),
            sample(&missing)
        ));
    }
    if !uncovered.is_empty() {
        e.push_str(&format!(
            "  checkpoint tensors NOT covered by the plan ({}):\n    {}\n",
            uncovered.len(),
            sample(&uncovered)
        ));
        e.push_str(
            "  -> this checkpoint uses weights this emitter does not implement.\n     \
             Compiling anyway would DROP them and emit a silently-wrong model.\n",
        );
    }
    Err(e)
}

fn bf16_round(f: f32) -> f32 {
    let u = f.to_bits();
    let r = u.wrapping_add(0x7fff).wrapping_add((u >> 16) & 1);
    f32::from_bits(r & 0xffff_0000)
}

/// The GEMM tiles the gfx950 kernels physically implement, each mapped to the DevOp that
/// selects it. `(op, BM, BN, BK)`. Every entry is a real, register-budgeted instantiation of
/// `d_gemm_t` in `runtime/amd/op_gemm.h` (all BK=64, 8-wave 2x4 grid), so the picker is only
/// ever allowed to name a tile the runtime can actually run. Add a kernel there, add a row here.
#[cfg(test)]
const GFX950_TILES: [(DevOp, u64, u64, u64); 3] = [
    (DevOp::Gemm, 256, 256, 64), // 144 KiB LDS — best when the shape SATURATES the 256 CUs
    (DevOp::GemmMed, 128, 128, 64), //  72 KiB LDS — fills the chip when a 256-tile leaves CUs idle
    (DevOp::GemmSmall, 64, 128, 64), // 55 KiB LDS — narrow M (short prompts / small chunks)
];

/// LDS the double-buffered A|B staging of a `BMxBNxBK` tile occupies, in bytes. Mirrors
/// `GM_LDS_HALVES_T` in `op_gemm.h`: 2 buffers x (BM+BN) rows x (BK+8 pad) halves x 2 B/half.
pub fn gemm_lds_bytes(bm: u64, bn: u64, bk: u64) -> u64 {
    2 * (bm + bn) * (bk + 8) * 2
}

/// Wall-clock cost of one GEMM tile for one shape — the single ranking used by
/// both `plowc tune` and the device-blob emitters (`pick_tile`).
///
/// Output tiles run in parallel, `n_units` at a time, so wall time is
/// `rounds x (cost of ONE tile)` and one tile costs `max(compute, dma)`: the
/// tile is double-buffered and SRAM-resident, so its operand fill hides behind
/// its matrix compute. Two opposing effects fall out with no hand-tuned
/// constants — a bigger tile has better arithmetic intensity `BM*BN/(BM+BN)`, a
/// smaller one makes more tiles and fills more units.
///
/// A tile whose working set overflows SRAM scores [`u64::MAX`] rather than being
/// dropped, so the ranking stays total. Ties resolve toward the larger tile via
/// a rank term in the low bits; without it, equal-cost shapes would resolve by
/// opcode number, and `GemmSmall` is 14 while `GemmMed` is 15.
pub fn tile_cost(
    spec: &hwspec::GpuSpec,
    kernel: &kernelcaps::KernelSpec,
    m: i64,
    n: i64,
    k: i64,
    n_units: u32,
) -> u64 {
    use costmodel::cost::{dma_cycles, macs_cycles};

    let Some(tile) = kernel.tile else { return u64::MAX };
    let (bm, bn, bk) = (tile.bm as u64, tile.bn as u64, tile.bk as u64);
    let (m, n, k) = (m.max(1) as u64, n.max(1) as u64, k.max(1) as u64);
    let n_units = (n_units as u64).max(1);

    if gemm_lds_bytes(bm, bn, bk) > spec.sm.shared_mem.0 {
        return u64::MAX;
    }

    let tiles = m.div_ceil(bm) * n.div_ceil(bn);
    let rounds = tiles.div_ceil(n_units);
    let k_iters = k.div_ceil(bk);
    let compute = k_iters * macs_cycles(spec, bm * bn * bk, hwspec::MmaDtype::Bf16);
    let dma = dma_cycles(spec, (bm * k + k * bn) * 2, false);
    let cost = rounds.saturating_mul(compute.max(dma));

    // Larger tile first on a tie: rank by descending BM*BN.
    let rank = match bm * bn {
        a if a >= 65536 => 0, // 256x256
        a if a >= 16384 => 1, // 128x128
        _ => 2,               // 64x128 and narrower
    };
    cost.saturating_mul(4).saturating_add(rank)
}

/// Pick the GEMM tile + inner-loop kernel for one `(M,N,K)` shape STATICALLY, from the gfx950
/// hardware spec — every shape in a `plow` schedule is known at compile time, so this is a
/// closed-form choice, not a runtime autotuner.
///
/// The choice is driven entirely by [`hwspec`] quantities, funnelled through the shared
/// [`costmodel`]: the bf16 MFMA rate (`sm.mma.bf16` x `tensor_cores`, via [`macs_cycles`]), the
/// HBM bandwidth (`mem.bandwidth`, via [`dma_cycles`]), the per-CU LDS budget
/// (`sm.shared_mem` = 160 KiB) and the CU count (`sm_count` = 256).
///
/// The model is WALL-CLOCK, not total-work — and that distinction is the whole point. Output
/// tiles run in PARALLEL, `n_cu` at a time, so wall time is `rounds x (cost of ONE tile)` where
/// `rounds = ceil(tiles / n_cu)`. One tile costs `max(compute, dma)`: the tile is double-buffered
/// and LDS-resident, so its HBM operand fill hides behind its MFMA compute. Two opposing effects
/// then fall straight out of the arithmetic, with no hand-tuned constants:
///
///   * a BIGGER tile has better arithmetic intensity (`BM*BN/(BM+BN)`), so lower per-tile DMA —
///     it wins once the shape SATURATES the CUs (q/o/gate/up/down at M=4096: >=256 tiles);
///   * a SMALLER tile makes MORE tiles, so it wins when the big tile leaves CUs idle — the
///     Llama/Qwen k/v projections (N=1024) are only 16x4 = 64 tiles at 256x256 (a quarter of the
///     256 CUs), and drop to a full 16x8 = 256 tiles at 128x128.
///
/// So `pick_tile` now returns `GemmMed` (128x128) for k/v and `Gemm` (256x256) for the wide
/// projections, filling all 256 CUs on both — where the old 3-candidate heuristic, blind to the
/// MFMA rate and to CU fill, pinned k/v to 256x256 and ran them on 64 CUs. It matches the
/// measured T=4096 optima (256x256 best) and generalises: Gemma-31B's kv_proj is N=4096, already
/// saturating, so it stays 256x256 — no regression.
fn pick_tile(m: u32, n: u32, k: u32, n_cu: u32) -> DevOp {
    let spec = hwspec::registry::lookup("MI350X").expect("gfx950 spec in registry");
    let hw = kernelcaps::HardwareFingerprint::from_spec(spec).expect("gfx950 fingerprint");
    let op = kernelcaps::OpSignature::gemm(kernelcaps::Phase::Prefill, m as i64, n as i64, k as i64);

    // The registry decides what is *executable*; the closure decides which of
    // those is fastest. Fusing both halves into one loop over a constant table
    // is what let this function name a tile the target does not implement
    // whenever it ran for a build that was not gfx950.
    let realization = kernelcaps::select_kernel(
        gfx950_gemm_inventory(),
        &op,
        &hw,
        kernelcaps::ProfileId::PrefillDense,
        &kernelcaps::NoMeasurements,
        |kernel| tile_cost(spec, kernel, m as i64, n as i64, k as i64, n_cu),
    )
    .expect("the gfx950 registry serves every prefill GEMM shape");

    realization.kernel.0
}

/// The gfx950 dense-GEMM inventory, derived by probing the interpreter object.
///
/// Probed when possible, analytical fallback otherwise. A hand-written tile
/// table is exactly what drifts from the object being compiled for, and AMD's
/// dispatch default silently no-ops an opcode with no arm
/// (`runtime/amd/interp.hip:785`), so drift would surface as slightly wrong
/// output rather than a crash. However, **requiring hipcc on a machine that
/// only targets NVIDIA** is a worse ergonomic failure than using known-stable
/// tile constants, so when the probe fails (hipcc missing) we fall back to
/// the analytical inventory — the same tiles the test fixture locks in.
#[cfg(not(test))]
fn gfx950_gemm_inventory() -> &'static kernelcaps::Inventory {
    use std::sync::OnceLock;
    static INV: OnceLock<kernelcaps::Inventory> = OnceLock::new();
    INV.get_or_init(|| {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        match kernelcaps::dense_gemm_inventory(&root, hwspec::IsaLevel::Gfx950) {
            Ok(inv) => inv,
            Err(e) => {
                eprintln!(
                    "warning: cannot probe gfx950 kernel inventory ({e}); \
                     using analytical fallback (known tile constants)"
                );
                gfx950_analytical_inventory()
            }
        }
    })
}

/// Analytical fallback inventory for gfx950 — the three tile instantiations
/// from `runtime/amd/op_gemm.h` (GM_BM/BN/BK, GM_MD_*, GM_SM_*). These are
/// compile-time constants in the interpreter object and change only with an
/// intentional ABI-breaking edit to op_gemm.h.
fn gfx950_analytical_inventory() -> kernelcaps::Inventory {
    use packet::dev::DevOp;
    let build = kernelcaps::BuildId::new(
        hwspec::IsaLevel::Gfx950,
        ["PLOW_BUCKET_DECODE=0".to_string()],
        "analytical-fallback",
        "analytical-fallback",
    );
    kernelcaps::Inventory::probed(
        build,
        [
            (DevOp::Gemm, 256, 256, 64, "gfx950:exec_gemm"),
            (DevOp::GemmMed, 128, 128, 64, "gfx950:exec_gemm_med"),
            (DevOp::GemmSmall, 64, 128, 64, "gfx950:exec_gemm_small"),
        ]
        .map(|(op, bm, bn, bk, body)| {
            kernelcaps::KernelSpec::gemm_tile(op, hwspec::IsaLevel::Gfx950, bm, bn, bk, body)
        }),
    )
}

/// Test fixture standing in for a probe.
///
/// This is a test *input*, not shipped data: it never reaches a compiled
/// artifact, and production has no path to it. It exists so the tile-selection
/// regression tests can run on a machine without ROCm, which is the only reason
/// the real probe is unavailable here.
#[cfg(test)]
fn gfx950_gemm_inventory() -> &'static kernelcaps::Inventory {
    use packet::dev::DevOp;
    use std::sync::OnceLock;
    static INV: OnceLock<kernelcaps::Inventory> = OnceLock::new();
    INV.get_or_init(|| {
        let build = kernelcaps::BuildId::new(
            hwspec::IsaLevel::Gfx950,
            ["PLOW_BUCKET_PREFILL=1".to_string()],
            "test-fixture",
            "test-fixture",
        );
        // The three instantiations in runtime/amd/op_gemm.h, with the GM_* tile
        // constants a probe would expand.
        kernelcaps::Inventory::probed(
            build,
            [
                (DevOp::Gemm, 256, 256, 64, "gfx950:exec_gemm"),
                (DevOp::GemmMed, 128, 128, 64, "gfx950:exec_gemm_med"),
                (DevOp::GemmSmall, 64, 128, 64, "gfx950:exec_gemm_small"),
            ]
            .map(|(op, bm, bn, bk, body)| {
                kernelcaps::KernelSpec::gemm_tile(op, hwspec::IsaLevel::Gfx950, bm, bn, bk, body)
            }),
        )
    })
}

fn tiles(m: u32, n: u32) -> u32 {
    m.div_ceil(BM) * n.div_ceil(BN)
}

/// Every tensor the model touches. Prefill and decode SHARE this table: the KV cache is
/// written by prefill and appended to by decode, and the 57 GiB of weights is not getting
/// loaded twice.
#[allow(dead_code)]
struct Tn {
    ids: u32,
    pos: u32,
    kvlen: u32,
    cos_s: u32,
    sin_s: u32,
    cos_f: u32,
    sin_f: u32,
    emb: u32,
    fin: u32,
    head: u32,
    // fp8 weight-only lm_head twin (PLOW_FP8_HEAD=1; tied models only). Separately-
    // labelled variant: vLLM fp8 keeps lm_head bf16, so report as its own row.
    head8: u32,
    head8s: u32,
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
    // TP (n_gpu>1) peer-mapped partial slots (plans/tp-design.md §7a): the row-parallel
    // o_proj/down write their partial H-vector here (tp-host binds them into peer_scratch),
    // and XReduce sums the N peers' slots into `og`/`dg`. TENSOR_NONE when tp==1.
    og_tp: u32,
    dg_tp: u32,
    kc: Vec<u32>,
    vc: Vec<u32>,
    // fp8-KV: per-(token,kv_head) f32 dequant scales, one per KV row (TENSOR_NONE in bf16 mode).
    kcs: Vec<u32>,
    vcs: Vec<u32>,
    // MoE decode scratch (shared across layers, B=1). dense MLP reuses fu/dg; the MoE branch adds:
    // h1 = post_ffn_norm_1(dense down); routing_table[k]; xn2 = pre_ffn_norm_2(residual);
    // mfu[k,I]; part[k,H] (f32); moe_sum[H]; h2 = post_ffn_norm_2(moe_sum); comb = h1+h2.
    moe_h1: u32,
    moe_tab: u32,
    moe_rscore: u32,
    moe_xn2: u32,
    moe_mfu: u32,
    moe_part: u32,
    moe_sum: u32,
    moe_h2: u32,
    moe_comb: u32,
    // Grouped-MoE PREFILL scratch (plans/p9-26b-prefill-moe.md). Declared only when moe && moe_pf;
    // TENSOR_NONE otherwise so the decode-only blob stays byte-identical. total_pad = rows*top_k +
    // n_exp*128 (PGM_BM). meta = int32[3*n_exp+2] align/sort table; rowtok/rowpart = u32[total_pad]
    // gather maps; rowgate = f32[total_pad]; fug = bf16[total_pad*moe_inter] gathered GLU output.
    moe_meta: u32,
    moe_rowtok: u32,
    moe_rowpart: u32,
    moe_rowgate: u32,
    moe_fug: u32,
    // beat26b w8a8 grouped-MoE prefill: fp8 twin of the gathered GLU output `fug` (uint8
    // [total_pad*moe_inter] e4m3 + f32 fscale[total_pad]), quantized by QuantFp8 between the w8a8
    // GLU and DOWN. TENSOR_NONE unless moe_pf && w8a8. (xn2 reuses xqh/ash — same hidden width.)
    moe_fuq: u32,
    moe_fus: u32,
    // T8 w8a8: reused-per-layer fp8 ACTIVATION quant scratch (uint8 xq + f32 row a_scale), one pair
    // per distinct activation width. Emitted only under PLOW_W8A8; TENSOR_NONE otherwise.
    //   xqh/ash  — hidden-width (q/k/v read n.hn; gate/up read n.hn again).
    //   xqo/aso  — qd-width (o_proj reads n.at).
    //   xqi/asi  — inter-width (down reads n.fu).
    // The three widths never alias in liveness within a layer (the DAG's existing edges serialize
    // qkv→flash→o→norm→gate/up→down), so each pair is reused across all 48 layers.
    xqh: u32,
    ash: u32,
    xqo: u32,
    aso: u32,
    xqi: u32,
    asi: u32,
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
    g_pf: u32,
    g_po: u32,
    qn: u32,
    kn: u32,
    // Gemma-4 MoE (26B-A4B) per-layer weights + the loader-filled expert pointer table.
    // TENSOR_NONE on the dense 12B/31B path. rproj/rscale/rpes = router; g_pf1/g_pf2/g_pre2 = the
    // three extra sandwich norms. ewt = Persistent u64[E*2] {gate_up base, down base} per expert,
    // filled by the harness/loader from the two FUSED expert tensors' bound bases. The fused expert
    // weights (experts.gate_up_proj [E,2I,H], experts.down_proj [E,H,I]) are declared as pkt tensors
    // (so the loader binds them by name) but are NOT op operands — the SM reaches them via ewt.
    rproj: u32,
    rscale: u32,
    rpes: u32,
    g_pf1: u32,
    g_pf2: u32,
    g_pre2: u32,
    ewt: u32,
    est: u32,
    // FP8 DECODE weights (PLOW_FP8) + their per-output-channel f32 dequant scales. The bf16 wq..wd
    // above stay bound (from the bf16 checkpoint) and feed PREFILL's GEMM; these fp8 twins feed the
    // decode GEMV. TENSOR_NONE in bf16 mode. The fp8 weight/scale tensors are declared under an
    // "fp8/" name prefix that the loader routes to the fp8 checkpoint (see gemma4_chat.c).
    wq8: u32,
    wk8: u32,
    wv8: u32,
    wo8: u32,
    wg8: u32,
    wu8: u32,
    wd8: u32,
    sq: u32,
    sk: u32,
    sv: u32,
    so: u32,
    sg: u32,
    su: u32,
    sd: u32,
}

fn declare(
    b: &mut Builder,
    c: &Cfg,
    ctx: u32,
    ns_pre: u32,
    fp8: bool,
    w8a8: bool,
    fp8_kv: bool,
    fp8_kv_full: bool,
    dbatch: u32,
    moe_pf: bool,
    block: std::ops::Range<usize>,
) -> Tn {
    // ACTIVATIONS ARE SIZED BY THE CHUNK, NOT THE CONTEXT.
    //
    // Every activation used to be `ctx * ...`, which is 131072 rows of scratch for a machine
    // that never has more than MAX_CHUNK=4096 rows in flight (prefill chunk) or 1 (decode).
    // That is a 32x over-allocation and it was 45.7 GiB of the 119 GiB footprint -- more than
    // the KV cache and nearly as much as the weights.
    //
    // Only `ids`/`pos` and the KV cache legitimately span the context: the cache IS the context,
    // and ids/pos are i32 (a rounding error). Everything else holds the CURRENT chunk.
    let rows = ctx.min(MAX_CHUNK);
    // TP head split (plans/tp-design.md §3a): each rank owns heads/N q-heads and kvh/N kv-heads,
    // so every head-dimensioned activation and the KV cache shrink by N. Column/row-parallel
    // weights and the inter/vocab-dimensioned activations shrink by N too. tp==1 => /1, identical.
    let tp = c.tp;
    assert_eq!(c.heads % tp, 0, "--tp {tp} must divide n_head {}", c.heads);
    assert_eq!(
        c.inter % tp,
        0,
        "--tp {tp} must divide intermediate {}",
        c.inter
    );
    // GEMV 8-wide load contract (runtime/nvidia/op_gemm.cuh): the decode GEMV family loads the
    // contraction dim (K) in 8-element vectors guarded only by `k < K`, so a K that is not a
    // multiple of 8 over-reads the final vector past the row. Every dim that becomes a GEMV K —
    // hidden (qkv/gate/up/lm_head), intermediate (down), and each head_dim (attn out) — must be
    // 8-aligned. Holds for all supported checkpoints; enforce it so an unaligned dim fails at
    // emit time instead of silently over-reading on device.
    assert_eq!(c.hidden % 8, 0, "hidden {} must be a multiple of 8 (GEMV 8-wide load)", c.hidden);
    assert_eq!(c.inter % 8, 0, "intermediate {} must be a multiple of 8 (GEMV 8-wide load)", c.inter);
    assert_eq!(c.hd_slide % 8, 0, "head_dim {} must be a multiple of 8 (GEMV 8-wide load)", c.hd_slide);
    assert_eq!(c.hd_full % 8, 0, "global_head_dim {} must be a multiple of 8 (GEMV 8-wide load)", c.hd_full);
    let qd_max = (c.heads / tp) * c.hd_slide.max(c.hd_full);
    // kv activation shards use the per-rank LOCAL kv-head count (shared-kv-head replication clamps
    // it to 1 when tp>kvh, so kvh/tp would under-size to 0 at tp=8 on full layers — §3a/§13.2).
    let kd_max =
        (kvh_local(c.kvh_slide, tp, 0) * c.hd_slide).max(kvh_local(c.kvh_full, tp, 0) * c.hd_full);
    let hd_max = c.hd_slide.max(c.hd_full);
    let inter_sh = c.inter / tp;
    // lm_head is REPLICATED under TP (Phase 2), not vocab-sharded. Two reasons the
    // sharded path is deferred: (1) Gemma ties lm_head to embed_tokens, and the emitted
    // lm_head Gemv reads `emb` from offset 0 with no per-rank vocab offset, so a vocab
    // shard would make every rank argmax the SAME low-vocab slice (silently wrong);
    // (2) XArgmaxFin (the cross-rank id-fold) is a stub. Replicating lm_head keeps the
    // full-vocab argmax correct on every rank (they agree), costs no extra memory (emb
    // is already fully resident for the embed lookup), and is one gemv/token — not the
    // decode bottleneck. Sharded lm_head + XArgmaxFin is a Phase-3 item (§8d, §13).
    let vocab_sh = c.vocab;
    let ac = |b: &mut Builder, n: &str, sz: u64| b.tensor(&format!("act.{n}"), sz);

    // RoPE tables are declared as RECIPES, not expanded bytes: at ctx=131072 the four
    // of them are ~403 MB, which dominated the blob, the load-time H2D, and nothing
    // else. The runtime materialises them from these scalars at bind time; `--no-rope-gen`
    // (Model::bake_gen) puts the bytes back for readers that predate v7.
    let [cs_s, sn_s] = GenTensor::rope_pair(ctx, c.hd_slide, c.theta_slide, 1.0, RopeScale::None);
    let [cs_f, sn_f] =
        GenTensor::rope_pair(ctx, c.hd_full, c.theta_full, c.rope_frac_full, c.rope_scale);

    // MoE row count for buffer sizing: 1 for the decode-only blob (byte-identical to the pre-prefill
    // path), the chunk `rows` when grouped-MoE prefill is enabled. total_pad bounds the token-sorted
    // gathered rows: rows*top_k routed slots + n_exp segments each padded up to the 128-row tile.
    let moe_pf_on = c.moe && moe_pf;
    // BATCH>1 DECODE: the decode MoE scratch is per-ROW ([B][k] table, [B][k][I] mfu,
    // [B][k][H] part, [B][n_exp] scores), so every per-token buffer is sized for B rows too.
    // dbatch==1 leaves every size exactly as it was => byte-identical blob.
    let moe_rows = (if moe_pf_on { rows } else { 1 }).max(dbatch);
    let total_pad = moe_rows * c.top_k + c.n_exp * 128;

    let t = Tn {
        ids: b.tensor("in.ids", ctx as u64 * I32),
        pos: b.tensor("in.pos", ctx as u64 * I32),
        // BATCH>1 (serving pending #4): one KV length per sequence. dbatch==1 => I32, identical.
        kvlen: b.tensor("in.kvlen", dbatch as u64 * I32),
        cos_s: b.tensor_gen("in.cos_slide", cs_s.byte_len(), cs_s),
        sin_s: b.tensor_gen("in.sin_slide", sn_s.byte_len(), sn_s),
        cos_f: b.tensor_gen("in.cos_full", cs_f.byte_len(), cs_f),
        sin_f: b.tensor_gen("in.sin_full", sn_f.byte_len(), sn_f),
        emb: b.tensor(
            &format!("{}embed_tokens.weight", c.prefix),
            (c.vocab * c.hidden) as u64 * BF16,
        ),
        fin: b.tensor(&format!("{}norm.weight", c.prefix), c.hidden as u64 * BF16),
        // Untied lm_head (Llama): a separate top-level "lm_head.weight". Tied models reuse emb.
        head: if c.tied {
            TENSOR_NONE
        } else {
            b.tensor("lm_head.weight", (c.vocab * c.hidden) as u64 * BF16)
        },
        head8: if c.tied && std::env::var("PLOW_FP8_HEAD").ok().as_deref() == Some("1") {
            b.tensor(
                &format!("fp8/{}embed_tokens.weight", c.prefix),
                (c.vocab * c.hidden) as u64,
            )
        } else {
            TENSOR_NONE
        },
        head8s: if c.tied && std::env::var("PLOW_FP8_HEAD").ok().as_deref() == Some("1") {
            b.tensor(
                &format!("fp8/{}embed_tokens.weight_scale", c.prefix),
                c.vocab as u64 * F32,
            )
        } else {
            TENSOR_NONE
        },
        x: ac(b, "x", (rows * c.hidden) as u64 * BF16),
        hn: ac(b, "hn", (rows * c.hidden) as u64 * BF16),
        qg: ac(b, "qg", (rows * qd_max) as u64 * BF16),
        kg: ac(b, "kg", (rows * kd_max) as u64 * BF16),
        vg: ac(b, "vg", (rows * kd_max) as u64 * BF16),
        q: ac(b, "q", (rows * qd_max) as u64 * BF16),
        // Sized for whichever phase needs more. Prefill: ctx * heads * ns_pre * hd.
        // Decode: 1 * heads * ns_dec * hd. Prefill wins for any sane ctx.
        // Sized as the MAX of what each phase needs, not as a product of both.
        //
        // It used to be `ctx * heads * ns_pre.max(8) * hd`. The `.max(8)` is there to cover the
        // DECODE program, whose nsplit is ~16 while prefill's is 1 at large T — but decode needs
        // only ONE row (`1 * heads * ns_dec * hd` = about 1 MB), and multiplying that 8x by CTX
        // is a 64 GiB over-allocation at ctx=128k. It is the difference between 239 GiB (does not
        // fit alongside 57 GiB of weights) and 183 GiB (does).
        // Head-split (heads/tp) attention partials.
        opart: ac(
            b,
            "opart",
            (rows.max(64) * (c.heads / tp) * ns_pre * hd_max).max((c.heads / tp) * 64 * hd_max)
                as u64
                * F32,
        ),
        mlpart: ac(
            b,
            "mlpart",
            (rows.max(64) * (c.heads / tp) * ns_pre * 2).max((c.heads / tp) * 64 * 2) as u64 * F32,
        ),
        at: ac(b, "at", (rows * qd_max) as u64 * BF16),
        og: ac(b, "og", (rows * c.hidden) as u64 * BF16),
        gt: ac(b, "gt", (rows * inter_sh) as u64 * BF16),
        ut: ac(b, "ut", (rows * inter_sh) as u64 * BF16),
        fu: ac(b, "fu", (rows * inter_sh) as u64 * BF16),
        dg: ac(b, "dg", (rows * c.hidden) as u64 * BF16),
        // Only the LAST row's logits are ever read in prefill (i4 = a_row0 on the lm_head), so
        // this is 512 KB, not the 2.1 GB a full-T lm_head would need at ctx=4096. Vocab-column-
        // sharded. BATCH>1 decode reads B rows (one per sequence), so *dbatch; dbatch==1 identical.
        logits: ac(b, "logits", (dbatch * vocab_sh) as u64 * BF16),
        // Per-block argmax partials, one packed u64 each. Needs no zeroing between steps:
        // every block writes its own slot unconditionally. BATCH>1: [dbatch][AMAX_BLOCKS].
        // E5 PLOW_FUSE_ARGMAX: the fused lm_head epilogue (GemvArgmax) runs on all n_cu blocks,
        // so it writes n_cu partials — size for max(AMAX_BLOCKS, n_cu). Gated on the flag, so the
        // default blob is byte-identical.
        amax: ac(
            b,
            "amax.part",
            dbatch as u64 * fuse_argmax_parts(b.n_cu()) as u64 * 8,
        ),
        // TP peer-mapped partials (§7a) — only declared under sharding; tp==1 leaves them absent
        // so the tensor table (and the whole blob) stays byte-identical to the pre-TP path.
        og_tp: if tp > 1 {
            ac(b, "og_tp", (rows * c.hidden) as u64 * BF16)
        } else {
            TENSOR_NONE
        },
        dg_tp: if tp > 1 {
            ac(b, "dg_tp", (rows * c.hidden) as u64 * BF16)
        } else {
            TENSOR_NONE
        },
        // MoE decode scratch (B=1). Only for the 26B-A4B MoE path; TENSOR_NONE otherwise so the
        // dense 12B/31B blob stays byte-identical. Sized by ONE token (decode); mfu/part are [k,·].
        // moe_rows scales the per-token MoE scratch (1 for decode, chunk `rows` for grouped prefill).
        moe_h1: if c.moe {
            ac(b, "moe.h1", (moe_rows * c.hidden) as u64 * BF16)
        } else {
            TENSOR_NONE
        },
        moe_tab: if c.moe {
            ac(b, "moe.table", (moe_rows * c.top_k) as u64 * 8)
        } else {
            TENSOR_NONE
        },
        moe_rscore: if c.moe {
            ac(b, "moe.router_score", (dbatch * c.n_exp) as u64 * F32)
        } else {
            TENSOR_NONE
        },
        moe_xn2: if c.moe {
            ac(b, "moe.xn2", (moe_rows * c.hidden) as u64 * BF16)
        } else {
            TENSOR_NONE
        },
        moe_mfu: if c.moe {
            ac(b, "moe.mfu", (dbatch * c.top_k * c.moe_inter) as u64 * BF16)
        } else {
            TENSOR_NONE
        },
        moe_part: if c.moe {
            ac(b, "moe.part", (moe_rows * c.top_k * c.hidden) as u64 * F32)
        } else {
            TENSOR_NONE
        },
        moe_sum: if c.moe {
            ac(b, "moe.sum", c.hidden as u64 * BF16)
        } else {
            TENSOR_NONE
        },
        moe_h2: if c.moe {
            ac(b, "moe.h2", c.hidden as u64 * BF16)
        } else {
            TENSOR_NONE
        },
        moe_comb: if c.moe {
            ac(b, "moe.comb", (moe_rows * c.hidden) as u64 * BF16)
        } else {
            TENSOR_NONE
        },
        // Grouped-MoE prefill scratch: declared only when moe && moe_pf. Appended AFTER every existing
        // tensor so the decode-only path's handles (and thus its packet bytes) are unchanged.
        moe_meta: if moe_pf_on {
            ac(b, "moe.meta", (3 * c.n_exp + 2) as u64 * I32)
        } else {
            TENSOR_NONE
        },
        moe_rowtok: if moe_pf_on {
            ac(b, "moe.rowtok", total_pad as u64 * I32)
        } else {
            TENSOR_NONE
        },
        moe_rowpart: if moe_pf_on {
            ac(b, "moe.rowpart", total_pad as u64 * I32)
        } else {
            TENSOR_NONE
        },
        moe_rowgate: if moe_pf_on {
            ac(b, "moe.rowgate", total_pad as u64 * F32)
        } else {
            TENSOR_NONE
        },
        moe_fug: if moe_pf_on {
            ac(b, "moe.fug", (total_pad * c.moe_inter) as u64 * BF16)
        } else {
            TENSOR_NONE
        },
        moe_fuq: if moe_pf_on && w8a8 {
            ac(b, "moe.fuq", (total_pad * c.moe_inter) as u64) // e4m3, 1 byte/elt
        } else {
            TENSOR_NONE
        },
        moe_fus: if moe_pf_on && w8a8 {
            ac(b, "moe.fus", total_pad as u64 * F32)
        } else {
            TENSOR_NONE
        },
        // T8 w8a8 fp8 activation-quant scratch (uint8 xq [rows*width] + f32 a_scale [rows]). One pair
        // per activation width, reused across all layers. TENSOR_NONE unless w8a8.
        xqh: if w8a8 {
            ac(b, "xqh", (rows * c.hidden) as u64)
        } else {
            TENSOR_NONE
        },
        ash: if w8a8 {
            ac(b, "ash", rows as u64 * F32)
        } else {
            TENSOR_NONE
        },
        xqo: if w8a8 {
            ac(b, "xqo", (rows * qd_max) as u64)
        } else {
            TENSOR_NONE
        },
        aso: if w8a8 {
            ac(b, "aso", rows as u64 * F32)
        } else {
            TENSOR_NONE
        },
        xqi: if w8a8 {
            ac(b, "xqi", (rows * inter_sh) as u64)
        } else {
            TENSOR_NONE
        },
        asi: if w8a8 {
            ac(b, "asi", rows as u64 * F32)
        } else {
            TENSOR_NONE
        },
        kc: Vec::new(),
        vc: Vec::new(),
        kcs: Vec::new(),
        vcs: Vec::new(),
        lw: Vec::new(),
    };
    let mut t = t;
    for l in 0..c.layers {
        // Block extraction: only in-range layers ALLOCATE per-layer tensors; the
        // rest push TENSOR_NONE so the Tn vectors stay FULL length (emit_phase
        // indexes them by absolute `l`). Full model => in_block always true =>
        // byte-identical allocation.
        let in_block = block.contains(&(l as usize));
        let full = c.is_full[l as usize];
        // MIXED fp8-KV (PLOW_FP8_KV_FULL=1, beat-fp8-mma): e4m3 cache on the hd512 FULL layers
        // only. Sliding rings are window-bounded (tiny), so fp8 buys them nothing; keeping them
        // bf16 keeps their shipped prefill/decode arms byte-identical and lets the fp8 PREFILL
        // object build PIPE=1 (the px4 fp8-mma arm is hd512-only).
        let fp8_kv = fp8_kv && (full || !fp8_kv_full);
        let hd = if full { c.hd_full } else { c.hd_slide };
        let kvh = if full { c.kvh_full } else { c.kvh_slide };
        // KV CACHE, HEAD-MAJOR [kv_head][ctx][hd] (see dev_isa.h "THE KV CACHE IS HEAD-MAJOR").
        // Exactly the layout HeadNormRope writes (op_norm.h d_headnorm_rope, out_stride = kvr)
        // and the layout FlashDecode reads with kv_stride — so the cache write is not a separate
        // copy, it IS the norm's store. Head-major makes one head's rows contiguous for the
        // decode read; a byte-repack (token-major, or vLLM-style paging) is a measured null here.
        // Per-layer head split with SHARED-KV-HEAD REPLICATION (plans/tp-design.md §3a/§13.2).
        // Full layers have kvh_full=4 kv-heads: tp<=4 splits cleanly (kvh_local = kvh/tp); at tp=8 a
        // full layer's 4 kv-heads can't split 8 ways, so tp/kvh ranks SHARE (replicate) each kv-head
        // — each such rank owns 1 kv-head (kvh_local=1) plus its heads/tp q-heads. KV storage is then
        // 2x on full layers only (a minority), the design's chosen tradeoff. Sliding layers (16 kv)
        // still split cleanly at tp=8. Requires kvh|tp OR tp|kvh; anything else fails loudly.
        let kvh_local = kvh_local(kvh, tp, l);
        let (kvr, _) = kv_ring(full, ctx);
        let qd = (c.heads / tp) * hd; // column-parallel q output shard
        let kd = kvh_local * hd; // column-parallel k/v output shard (KV head-sharded/replicated)
                                 // fp8-KV: the cache is uint8 e4m3 (1 byte/elem, HALF the bf16 footprint) plus a per-row
                                 // f32 scale [kv_head][ctx] (head-major, same RING as the cache). Written by HeadNormRopeFp8,
                                 // read by FlashDecodeFp8 / FlashPrefillFp8. bf16 mode keeps the 2-byte cache and no scales.
        let kv_elt = if fp8_kv { 1 } else { BF16 };
        // BATCH>1 (serving pending #4): the KV cache is BATCH-MAJOR [dbatch][kv_head][ring][hd] —
        // each sequence owns its own ring. d_flash_decode/d_headnorm_rope index it per-batch as
        // ((b*n_kv_head+hkv)*kv_stride+row)*hd, so the per-batch stride is kv_head*ring*hd and the
        // tensor is dbatch* that. dbatch==1 => byte-identical to the single-sequence cache.
        let db = dbatch as u64;
        t.kc.push(if in_block {
            b.tensor(&format!("kv.{l}.k"), db * (kvr * kvh_local * hd) as u64 * kv_elt)
        } else {
            TENSOR_NONE
        });
        t.vc.push(if in_block {
            b.tensor(&format!("kv.{l}.v"), db * (kvr * kvh_local * hd) as u64 * kv_elt)
        } else {
            TENSOR_NONE
        });
        t.kcs.push(if fp8_kv && in_block {
            b.tensor(
                &format!("kv.{l}.k_scale"),
                db * (kvr * kvh_local) as u64 * F32,
            )
        } else {
            TENSOR_NONE
        });
        t.vcs.push(if fp8_kv && in_block {
            b.tensor(
                &format!("kv.{l}.v_scale"),
                db * (kvr * kvh_local) as u64 * F32,
            )
        } else {
            TENSOR_NONE
        });
        let prefix = c.prefix.clone();
        // All per-layer weight declarations funnel through these closures; gating
        // them on `in_block` drops out-of-range layers' weights from the tensor
        // table (the loader binds nothing for them). Full model => always alloc.
        let w = |b: &mut Builder, s: &str, sz: u64| {
            if in_block {
                b.tensor(&format!("{prefix}layers.{l}.{s}"), sz)
            } else {
                TENSOR_NONE
            }
        };
        // T6 L2: in fp8 mode BOTH prefill (GemmFp8) and decode (GemvFp8) consume the fp8 twins, so
        // the bf16 projection weight is DEAD — declaring it still made the loader stream 22 GiB of
        // never-read weight (fp8 pkt was 32.3 GiB = 22.2 bf16 + 10.1 fp8). Elide the bf16 projection
        // in fp8 mode (norms, embedding/lm_head, RoPE stay bf16). Verified: every w.wq..wd reference
        // (fused GemvQkv, bf16 GemmGlu/GemvGlu, bf16 proj arm) is under a `!fp8` guard.
        let wproj = |b: &mut Builder, s: &str, sz: u64| {
            if fp8 || !in_block {
                TENSOR_NONE
            } else {
                b.tensor(&format!("{prefix}layers.{l}.{s}"), sz)
            }
        };
        // A weight that only some architectures ship: declared only when present, else NONE — so
        // the runtime never tries to bind a tensor the checkpoint does not have.
        let wopt = |b: &mut Builder, present: bool, s: &str, sz: u64| {
            if present && in_block {
                b.tensor(&format!("{prefix}layers.{l}.{s}"), sz)
            } else {
                TENSOR_NONE
            }
        };
        let keqv = full && c.k_eq_v;
        let gemma = c.arch == Arch::Gemma4;
        // MoE fused expert weights: declared so the loader binds them by name and the harness
        // derives per-expert ewt bases from their device addresses. Not referenced as op operands
        // (the SM indexes them through the ewt pointer table), so the handles are discarded here.
        if c.moe && in_block {
            let gu_n = (c.n_exp * 2 * c.moe_inter) as u64;
            let dn_n = (c.n_exp * c.hidden) as u64;
            if fp8 {
                b.tensor(
                    &format!("fp8/{prefix}layers.{l}.experts.gate_up_proj"),
                    gu_n * c.hidden as u64,
                );
                b.tensor(
                    &format!("fp8/{prefix}layers.{l}.experts.gate_up_proj_scale"),
                    gu_n * F32,
                );
                b.tensor(
                    &format!("fp8/{prefix}layers.{l}.experts.down_proj"),
                    dn_n * c.moe_inter as u64,
                );
                b.tensor(
                    &format!("fp8/{prefix}layers.{l}.experts.down_proj_scale"),
                    dn_n * F32,
                );
            } else {
                w(b, "experts.gate_up_proj", gu_n * c.hidden as u64 * BF16);
                w(b, "experts.down_proj", dn_n * c.moe_inter as u64 * BF16);
            }
        }
        // FP8 decode twin of a projection: the quantized weight (1 byte/elt) under an "fp8/" name
        // the loader routes to the fp8 checkpoint, plus its per-output-channel f32 dequant scale
        // ("<name>_scale", [out]). `out` is the row count of the [out,in] weight = numel/in.
        let w8 = |b: &mut Builder, s: &str, numel: u64| -> u32 {
            if fp8 && in_block {
                b.tensor(&format!("fp8/{prefix}layers.{l}.{s}"), numel)
            } else {
                TENSOR_NONE
            }
        };
        let sc = |b: &mut Builder, s: &str, out: u64| -> u32 {
            if fp8 && in_block {
                b.tensor(&format!("fp8/{prefix}layers.{l}.{s}_scale"), out * F32)
            } else {
                TENSOR_NONE
            }
        };
        t.lw.push(LW {
            wq: wproj(b, "self_attn.q_proj.weight", (qd * c.hidden) as u64 * BF16),
            wk: wproj(b, "self_attn.k_proj.weight", (kd * c.hidden) as u64 * BF16),
            // Gemma full layers have NO v_proj: V is the raw k_proj output (k_eq_v). Llama/Qwen
            // always have a real v_proj. (fp8 mode elides the bf16 twin like the other projections.)
            wv: wopt(
                b,
                !keqv && !fp8,
                "self_attn.v_proj.weight",
                (kd * c.hidden) as u64 * BF16,
            ),
            wo: wproj(b, "self_attn.o_proj.weight", (c.hidden * qd) as u64 * BF16),
            wg: wproj(
                b,
                "mlp.gate_proj.weight",
                (inter_sh * c.hidden) as u64 * BF16,
            ),
            wu: wproj(b, "mlp.up_proj.weight", (inter_sh * c.hidden) as u64 * BF16),
            wd: wproj(
                b,
                "mlp.down_proj.weight",
                (c.hidden * inter_sh) as u64 * BF16,
            ),
            // fp8 twins (numel bytes) + scales ([out] f32). k_eq_v layers have no v_proj to quantize.
            // Dims use the TP-sharded shard extents (qd/kd/inter_sh); at tp==1 these equal the full
            // extents, so the single-GPU fp8 pkt is unaffected by the TP structure.
            wq8: w8(b, "self_attn.q_proj.weight", (qd * c.hidden) as u64),
            wk8: w8(b, "self_attn.k_proj.weight", (kd * c.hidden) as u64),
            wv8: if keqv {
                TENSOR_NONE
            } else {
                w8(b, "self_attn.v_proj.weight", (kd * c.hidden) as u64)
            },
            wo8: w8(b, "self_attn.o_proj.weight", (c.hidden * qd) as u64),
            wg8: w8(b, "mlp.gate_proj.weight", (inter_sh * c.hidden) as u64),
            wu8: w8(b, "mlp.up_proj.weight", (inter_sh * c.hidden) as u64),
            wd8: w8(b, "mlp.down_proj.weight", (c.hidden * inter_sh) as u64),
            sq: sc(b, "self_attn.q_proj.weight", qd as u64),
            sk: sc(b, "self_attn.k_proj.weight", kd as u64),
            sv: if keqv {
                TENSOR_NONE
            } else {
                sc(b, "self_attn.v_proj.weight", kd as u64)
            },
            so: sc(b, "self_attn.o_proj.weight", c.hidden as u64),
            sg: sc(b, "mlp.gate_proj.weight", inter_sh as u64),
            su: sc(b, "mlp.up_proj.weight", inter_sh as u64),
            sd: sc(b, "mlp.down_proj.weight", c.hidden as u64),
            g_in: w(b, "input_layernorm.weight", c.hidden as u64 * BF16),
            g_pa: w(b, "post_attention_layernorm.weight", c.hidden as u64 * BF16),
            // Gemma's sandwich has two extra norms; Llama/Qwen do not.
            g_pf: wopt(
                b,
                gemma,
                "pre_feedforward_layernorm.weight",
                c.hidden as u64 * BF16,
            ),
            g_po: wopt(
                b,
                gemma,
                "post_feedforward_layernorm.weight",
                c.hidden as u64 * BF16,
            ),
            qn: wopt(
                b,
                c.has_qk_norm,
                "self_attn.q_norm.weight",
                hd as u64 * BF16,
            ),
            kn: wopt(
                b,
                c.has_qk_norm,
                "self_attn.k_norm.weight",
                hd as u64 * BF16,
            ),
            // MoE (26B-A4B): router + FUSED 3D expert weights + the 3 extra sandwich norms. The
            // ewt pointer table is NOT a checkpoint tensor — it is a Persistent buffer the harness/
            // loader fills with per-expert bases derived from the two fused tensors' devp bases.
            rproj: wopt(
                b,
                c.moe,
                "router.proj.weight",
                (c.n_exp * c.hidden) as u64 * BF16,
            ),
            rscale: wopt(b, c.moe, "router.scale", c.hidden as u64 * BF16),
            rpes: wopt(b, c.moe, "router.per_expert_scale", c.n_exp as u64 * BF16),
            g_pf1: wopt(
                b,
                c.moe,
                "post_feedforward_layernorm_1.weight",
                c.hidden as u64 * BF16,
            ),
            g_pf2: wopt(
                b,
                c.moe,
                "post_feedforward_layernorm_2.weight",
                c.hidden as u64 * BF16,
            ),
            g_pre2: wopt(
                b,
                c.moe,
                "pre_feedforward_layernorm_2.weight",
                c.hidden as u64 * BF16,
            ),
            ewt: if c.moe && in_block {
                b.tensor(&format!("moe.ewt.{l}"), (c.n_exp * 2) as u64 * 8)
            } else {
                TENSOR_NONE
            },
            est: if c.moe && fp8 && in_block {
                b.tensor(&format!("moe.est.{l}"), (c.n_exp * 2) as u64 * 8)
            } else {
                TENSOR_NONE
            },
        });
    }
    t
}

const Q_TILE_ROWS: u32 = 8 * 32; // PLOW_WAVES * FA_BQ — keep in step with amd_common.h

/// LDS the GEMM arena holds, in halves. Mirrors `GM_LDS_HALVES` in `op_gemm.h`:
/// `2*(GM_BM+GM_BN)*(GM_BK+8)` = `2*(256+256)*72`. A GEMV can stage its A-operand on-chip only
/// if `M*K` fits here, which [`DevOp::GemvGlu`] requires (it re-reads x per output column).
const GM_LDS_HALVES: u64 = 2 * (256 + 256) * (64 + 8);

/// Largest prefill chunk. Mirrors `PLOW_MAX_CHUNK` in `dev_isa.h`.
///
/// This is the ONLY row count any single program ever processes: chunked prefill never emits a
/// chunk bigger than this, and decode is one row. So it caps BOTH the bucket ladder (a program
/// for T > MAX_CHUNK can never be invoked) and every ACTIVATION tensor (they hold the current
/// chunk, not the context -- only the KV cache spans the context).
const MAX_CHUNK: u32 = 8192;

/// SLIDING-WINDOW KV RING. Mirrors `PLOW_KV_RING` / `PLOW_KV_MASK_NONE` in `dev_isa.h`.
const KV_RING: u32 = 16384;
const KV_MASK_NONE: u32 = 0xFFFF_FFFF;

/// How many rows a layer's KV cache actually needs, and the mask its row index is ANDed with.
///
/// A sliding layer never looks back further than `window`, so it needs a RING rather than the
/// full context — at ctx=128k that is 100 GiB of never-read cache. A full-attention layer keeps
/// a linear cache and gets `0xFFFFFFFF`, so the AND in the kernels is a no-op there.
///
/// The ring must be at least `window + max_chunk - 1`: a prefill chunk's queries span
/// `[c0, c0+C)` and between them read `[c0-window+1, c0+C-1]`, and the chunk writes all C of its
/// rows before flash reads any of them. See `PLOW_KV_RING` in dev_isa.h. It is a power of two so
/// the kernels can AND rather than divide.
fn kv_ring(full: bool, ctx: u32) -> (u32, u32) {
    if full {
        (ctx, KV_MASK_NONE)
    } else {
        let r = ctx.min(KV_RING); // no point ringing a cache smaller than the ring
        // `row & (r-1)` is a modulo ONLY when r is a power of two. For a non-pow2 r the AND
        // aliases rows to WRONG (in-bounds) rows — silent corruption. All shipped ctx are
        // pow2; make the invariant loud (leak-audit finding #6).
        assert!(r.is_power_of_two(), "kv_ring size {r} (ctx {ctx}) must be a power of two");
        (r, r - 1)
    }
}

/// This rank's local KV-head count under TP with SHARED-KV-HEAD REPLICATION (plans/tp-design.md
/// §3a/§13.2). Two regimes, both keep every rank's q-heads mapped to a kv-head it owns:
///   - `tp <= kvh_g` (clean split): each rank owns `kvh_g/tp` distinct kv-heads.
///   - `tp  > kvh_g` (replication): `tp/kvh_g` ranks share (replicate) one kv-head; each owns 1.
/// Anything else (neither divides) is unsupported and fails loudly rather than shard silently wrong.
fn kvh_local(kvh_g: u32, tp: u32, l: u32) -> u32 {
    if tp <= kvh_g {
        assert_eq!(
            kvh_g % tp,
            0,
            "--tp {tp} must divide layer {l}'s kv-heads {kvh_g} (§3a/§13.2)"
        );
        kvh_g / tp
    } else {
        assert_eq!(
            tp % kvh_g,
            0,
            "--tp {tp} must be a multiple of layer {l}'s kv-heads {kvh_g} for shared-kv-head \
             replication (§3a/§13.2)"
        );
        1
    }
}

/// Which `d_gemv` workgroups produce output columns `[c0, c1)`?
///
/// Requires `GV_BLOCKED=1` in `op_gemm.h`, where workgroup `s` owns the contiguous run
/// `[s*per, s*per+per)`, `per = ceil(N/nblk)`. Under the DEFAULT interleaved assignment this
/// function would be a lie: a workgroup's columns are `[8s, 8s+8) (mod nblk*8)`, scattered
/// across all of N, so 256 consecutive columns touch EVERY workgroup and the answer is always
/// "all of them" (measured: 128 of 128).
fn gemv_wgs_for_cols(n: u32, nblk: u32, c0: u32, c1: u32) -> Vec<u32> {
    let per = n.div_ceil(nblk);
    (c0 / per..=(c1 - 1) / per).filter(|&w| w < nblk).collect()
}

/// The work items (`token * nhead + head`) that `headnorm_rope` workgroup `j` runs.
///
/// `d_headnorm_rope` walks `for (w = slice*PLOW_WAVES + wave; w < total; w += nblk*PLOW_WAVES)`,
/// so workgroup `j` owns the items whose wave slot lands in `[8j, 8j+8)`.
fn headnorm_items(nblk: u32, total: u32, j: u32) -> Vec<u32> {
    (0..total)
        .filter(|&w| (w % (nblk * WAVES)) / WAVES == j)
        .collect()
}

/// The headnorm workgroup that produces item `w`.
fn headnorm_wg_of(nblk: u32, w: u32) -> u32 {
    (w % (nblk * WAVES)) / WAVES
}

const WAVES: u32 = 8; // PLOW_WAVES

/// MoE GLU → Down dependency map: Down block `b` only needs the GLU blocks that produce
/// the slots it reads. Without this, Down waits for ALL GLU blocks (coarse gate), wasting
/// ~2.6M cycles per layer on the critical path.
///
/// Layout: GLU produces flat `[k * I_moe]` outputs distributed round-robin across `nblk`
/// blocks (per_g = ceil(k*I_moe/nblk) per block). Down produces flat `[k * H]` outputs
/// similarly (per_d = ceil(k*H/nblk)). Down block `b` handles flat indices
/// `[b*per_d, (b+1)*per_d)`. For flat index `f`, `slot = f / H`. Down reads
/// `fu[slot*I_moe..(slot+1)*I_moe]` — so it depends on GLU blocks covering that range.
fn moe_down_fine_map(top_k: u32, i_moe: u32, hidden: u32, nblk: u32) -> Vec<Vec<u32>> {
    let total_glu = top_k * i_moe;
    let total_down = top_k * hidden;
    let per_g = total_glu.div_ceil(nblk);
    let per_d = total_down.div_ceil(nblk);
    (0..nblk)
        .map(|b| {
            let f0 = b * per_d;
            let f1 = ((b + 1) * per_d).min(total_down);
            if f0 >= total_down {
                return vec![];
            }
            let slot_lo = f0 / hidden;
            let slot_hi = (f1 - 1) / hidden;
            let glu_lo = slot_lo * i_moe;
            let glu_hi = (slot_hi + 1) * i_moe;
            let g_first = glu_lo / per_g;
            let g_last = (glu_hi - 1) / per_g;
            (g_first..=g_last.min(nblk - 1)).collect()
        })
        .collect()
}

/// The flash → merge edge is SPARSE, and today it is gated as if it were dense.
///
/// `flash_*` splits its work into `q_tiles * n_head * nsplit` items, item
/// `w = (qt * n_head + h) * nsplit + sp`, run by workgroup `w % nblk_f` (the kernels walk
/// `for (w = slice; w < n_work; w += nblk)`). `flash_merge` splits into `n_bh = n_batch *
/// n_head` items, item `w = b * n_head + h`, run by workgroup `w % nblk_m`. Merge item
/// `(b, h)` folds the `nsplit` partials of that same `(b, h)` and touches nothing else.
///
/// So a merge workgroup needs a handful of flash slices — at Gemma-31B decode, **8 of 256**.
/// Coarse counters make it wait for all 256, and the trace says that wait costs 0.83 ms of a
/// 16.9 ms token: the gate opens on the slowest CU, and 256 CUs doing this work spread over
/// 9.6-16.6 us.
///
/// `rows_per_item` is how many query rows one flash work item covers: `Q_TILE_ROWS` in
/// prefill (flash tiles the q axis) and 1 in decode (there is one query row).
fn flash_merge_map(
    n_bh: u32,
    nsplit: u32,
    rows_per_item: u32,
    n_head: u32,
    nblk_f: u32,
    nblk_m: u32,
) -> Vec<Vec<u32>> {
    (0..nblk_m)
        .map(|j| {
            let mut s: Vec<u32> = (0..n_bh)
                .filter(|w| w % nblk_m == j) // the merge items THIS workgroup runs
                .flat_map(|w| {
                    let (b, h) = (w / n_head, w % n_head);
                    let qt = b / rows_per_item; // which flash q-tile covers this row
                    (0..nsplit).map(move |sp| ((qt * n_head + h) * nsplit + sp) % nblk_f)
                })
                .collect();
            s.sort_unstable();
            s.dedup();
            s
        })
        .collect()
}

/// Emit the layer all-reduce for a row-parallel producer (o_proj/down), all-reduce #1/#2.
/// PREFILL uses the TWO-SHOT (reduce-scatter + all-gather): the [T,hidden] partial is
/// bandwidth-bound, so ~N/2× less fabric than the one-shot (plans/tp-prefill.md §4). DECODE
/// keeps the one-shot — its tiny [1,hidden] message is latency-bound, so a single sync wins.
/// Two-shot consumes TWO xctr gate ids (reduce-scatter + all-gather rendezvous); one-shot
/// consumes one. `slot` is the byte offset of this collective's partial slot (0 or slot_b).
/// Result is BIT-IDENTICAL across the two variants (same f32-acc, r=0..N−1 order).
#[allow(clippy::too_many_arguments)]
fn emit_xreduce(
    b: &mut Builder,
    xgate: &mut u32,
    decode: bool,
    xr_cus: &[u32],
    dep: u32,
    out: u32,
    xr_elems: u32,
    tp: u32,
    slot: u32,
) -> u32 {
    if decode {
        let gate = *xgate;
        *xgate += 1;
        b.emit(DevOp::XReduce, xr_cus.to_vec(), &[dep], |d| {
            d.t[0] = out; // reduced [1,hidden] result (local)
            d.i[0] = xr_elems; // elements to reduce (decode: hidden)
            d.i[1] = tp; // n_gpu
            d.i[2] = slot; // partial slot byte offset (§7a)
            d.i[3] = gate; // xctr gate id (unique per collective)
        })
    } else {
        let gate_rs = *xgate;
        *xgate += 1;
        let gate_ag = *xgate;
        *xgate += 1;
        b.emit(DevOp::XReduceTwoShot, xr_cus.to_vec(), &[dep], |d| {
            d.t[0] = out; // reduced [t,hidden] result (local)
            d.i[0] = xr_elems; // elements to reduce (t*hidden)
            d.i[1] = tp; // n_gpu
            d.i[2] = slot; // partial slot byte offset (§7a)
            d.i[3] = gate_rs; // reduce-scatter rendezvous gate id
            d.i[4] = gate_ag; // all-gather rendezvous gate id
        })
    }
}

/// Which program `emit_phase` is building. This used to be a `decode: bool`, which conflated two
/// INDEPENDENT axes — and the whole point of the enum is to pull them apart:
///
///   * **shape** — one query row, KV *append* + ring mask, decode's nsplit. (`decode_shape`)
///   * **kernel family** — the GEMV opcodes and the fusions that only exist because of them
///     (fold / fuse_norm / gfuse / fuse_qkv / glu_fused), plus flash-DECODE attention. (`gemv`)
///
/// `Decode` is (shape, gemv) = (true, true) and `Prefill` is (false, false) — the only two
/// combinations that existed before — so both stay BYTE-IDENTICAL to the pre-enum emitter.
/// `DecodeTiled` is the new third corner: (true, false), a decode-shaped step built from prefill
/// kernels. See plans/decode-tiled.md.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Mode {
    Prefill,
    Decode,
    /// Decode shape, prefill kernels: tiled GEMM + FlashPrefill at one query row. Targets long
    /// context, where GEMV does not scale with batch (its split is over N, not M) and FlashDecode
    /// caps at n_cu. **Requires prefill opcodes in the interpreter** — the sm_120 build traps on
    /// FlashPrefill(11)/GemmSmall(14)/GemmMed(15)/GemmGlu(20), so this mode is AMD-only today.
    DecodeTiled,
}

impl Mode {
    /// One query row, KV append + ring mask, decode's nsplit and one-shot all-reduce.
    fn decode_shape(self) -> bool {
        self != Mode::Prefill
    }
    /// The GEMV opcode family and every fusion that exists only to serve it, plus flash-decode.
    fn gemv(self) -> bool {
        self == Mode::Decode
    }
}

/// Split router is DEFAULT-ON: the 128-expert score GEMV runs on 16 CTAs (8 experts/CTA)
/// instead of serializing on one CTA. The fused single-CTA path is the escape hatch.
fn gemma_moe_router_split_enabled() -> bool {
    std::env::var("GLM_ROUTER_OLD").ok().as_deref() != Some("1")
        && std::env::var("PLOW_GEMMA_MOE_ROUTER_FUSED").ok().as_deref() != Some("1")
}

/// `nrow` = decode batch B: the score work space is the (row, expert) PAIR space, so B rows
/// scale the useful CTA count (16 CTAs at B=1/E=128, capped at n_cu from B=12 up).
fn gemma_moe_router_split_plan(n_cu: u32, n_exp: u32, nrow: u32) -> Option<(u32, DevOp)> {
    if !gemma_moe_router_split_enabled() {
        return None;
    }
    let max_useful = (nrow * n_exp).div_ceil(8).max(1).min(n_cu.max(1));
    let blocks = std::env::var("PLOW_GEMMA_MOE_ROUTER_BLOCKS")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(max_useful)
            .clamp(1, max_useful);
    let op = if std::env::var("PLOW_GEMMA_MOE_ROUTER_EXACT").ok().as_deref() == Some("1") {
        DevOp::MoeRouterGemmaScore
    } else {
        DevOp::MoeRouterGemmaScoreFast
    };
    Some((blocks, op))
}

#[allow(clippy::too_many_arguments)]
fn emit_gemma_moe_router(
    b: &mut Builder,
    dep: u32,
    table: u32,
    resid: u32,
    proj: u32,
    scale: u32,
    pes: u32,
    score: u32,
    hidden: u32,
    n_exp: u32,
    top_k: u32,
    root: f32,
    eps: f32,
    split_plan: Option<(u32, DevOp)>,
    nrow: u32,
) -> u32 {
    // BATCH B>1: the batch row count rides a spare immediate, emitted ONLY when B>1 so the
    // B=1 instruction bytes are untouched (the kernels read 0 as "one row").
    let nb = if nrow > 1 { nrow } else { 0 };
    if let Some((blocks, score_op)) = split_plan {
        assert_ne!(
            score, TENSOR_NONE,
            "split Gemma router requires f32 score scratch"
        );
        let c_score = b.emit(
            score_op,
            (0..blocks).collect(),
            &[dep],
            |d| {
                d.t[0] = score;
                d.t[1] = resid;
                d.t[2] = proj;
                d.t[3] = scale;
                d.i[0] = hidden;
                d.i[1] = n_exp;
                d.i[2] = nb;
                d.f[0] = root;
                d.f[1] = eps;
            },
        );
        // top-k is serial per row; give it one CTA per row so B rows run concurrently.
        let topk_cus: Vec<u32> = (0..nrow.max(1)).collect();
        b.emit(DevOp::MoeRouterGemmaTopk, topk_cus, &[c_score], |d| {
            d.t[0] = table;
            d.t[1] = score;
            d.t[2] = pes;
            d.i[1] = n_exp;
            d.i[2] = top_k;
            d.i[3] = nb;
        })
    } else {
        b.emit(DevOp::MoeRouterGemma, vec![0], &[dep], |d| {
            d.t[0] = table;
            d.t[1] = resid;
            d.t[2] = proj;
            d.t[3] = scale;
            d.t[4] = pes;
            d.i[0] = hidden;
            d.i[1] = n_exp;
            d.i[2] = top_k;
            d.i[3] = nb;
            d.f[0] = root;
            d.f[1] = eps;
        })
    }
}

/// Emit one phase. `t == 1 && decode` is the decode step; otherwise a prefill bucket.
fn emit_phase(
    b: &mut Builder,
    c: &Cfg,
    ls: &[f32],
    n: &Tn,
    t: u32,
    ctx: u32,
    mode: Mode,
    n_cu: u32,
    kv_rows: &mut Vec<u32>,
    fp8: bool,
    w8a8: bool,
    fp8_kv: bool,
    fp8_kv_full: bool,
    block: std::ops::Range<usize>,
    block_mode: bool,
) {
    // The two axes the old `decode` bool used to carry at once. Every former use site below is
    // now one or the other: `decode` for shape, `gemv_family` for kernel family. (Not `gemv` —
    // the `hn_dep` closure below already binds a `gemv: u32` parameter that would shadow it.)
    let decode = mode.decode_shape();
    let gemv_family = mode.gemv();
    let all = b.all();
    // TENSOR-PARALLEL local shards (plans/tp-design.md §3). For tp==1 these equal the full dims,
    // so the whole emit is byte-identical to the pre-TP path; for tp>1 (decode only) every head-,
    // intermediate- and vocab-dimensioned op runs 1/N wide, and o_proj/down get an XReduce.
    let tp = c.tp;
    let heads = c.heads / tp; // this rank's q-heads
    let inter_l = c.inter / tp; // this rank's gate/up/down intermediate lanes
    let vocab_l = c.vocab; // lm_head REPLICATED under TP (Phase 2); see declare() note above
    let mut xgate: u32 = 0; // xctr gate-id allocator for XReduce (unique per collective)
                            // XReduce runs on a REDUCED CU set (F-lever, plans/tp-design.md §8b/§10). The all-reduce is a
                            // tiny memory-bound sum over the H-vector, but EVERY participating workgroup takes a SYSTEM-scope
                            // acquire (a full L2 invalidate) per collective — 2L=120 collectives/token at 31B. Fewer CUs =>
                            // fewer redundant system-acquires and less cross-XCD invalidation, at no bandwidth cost (H=5376
                            // saturates on a handful of workgroups). Default keeps `all` (byte-identical to Phase-2); set
                            // PLOW_XR_CUS=k to cap it (measured lever for the TP=8 NUMA-crossing all-reduce). tp==1 unused.
    let xr_cus: Vec<u32> = {
        let k = std::env::var("PLOW_XR_CUS")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(n_cu)
            .clamp(1, n_cu);
        (0..k).collect()
    };
    // TP prefill (plans/tp-prefill.md): the all-reduce partials are [T, hidden], not decode's
    // [1, hidden], so the XReduce reduces `xr_elems = t*hidden` elements. The two peer-scratch
    // partial slots (og_tp/dg_tp, §7a) must not overlap: partial_A occupies [0, rows_max*hidden*2),
    // partial_B starts at `slot_b = rows_max*hidden*2`. rows_max = the largest chunk (= og_tp's
    // declared row count in declare()), so the slot is IDENTICAL across every prefill bucket AND
    // the decode program — the host binds dg_tp at that one fixed offset for all of them. For
    // decode t==1 so xr_elems==hidden and the layout is a superset of the old decode path.
    let rows_max = ctx.min(MAX_CHUNK);
    let xr_elems = t * c.hidden;
    let slot_b = rows_max * c.hidden * BF16 as u32;
    let rows: Vec<u32> = (0..t.min(n_cu).max(1)).collect();
    // Elementwise ops sized to their ACTUAL work, not handed the whole machine.
    //
    // A decode residual is 5376 elements. On 256 CUs that is 21 elements each -- the op is
    // pure gate overhead, and all 256 workgroups still have to be counted into the barrier.
    // One workgroup (512 threads x 8) covers it. Fewer participants, cheaper gate, less
    // counter contention.
    let elem = |n: u32| -> Vec<u32> { (0..n.div_ceil(512 * 8).max(1).min(n_cu)).collect() };
    let ns = if gemv_family {
        n_cu.div_ceil(heads).max(1)
    } else {
        n_cu.div_ceil((t.div_ceil(Q_TILE_ROWS) * heads).max(1))
            .max(1)
    };
    // When nsplit==1 there is nothing for d_flash_merge to combine: flash_prefill normalizes
    // in its own epilogue and writes the final bf16 straight to n.at, and the merge op is not
    // emitted at all. Prefill-only (decode always keeps ns>1).
    let fused = !gemv_family && ns == 1;

    let escale = c.emb_scale;
    // Block mode: no token embedding — `act.x` is uploaded by the harness (the
    // residual-stream input), so Embed would overwrite it. The first in-block
    // layer's RmsNorm reads `act.x` directly (its dep is `&[]`; see below).
    let mut dep = if block_mode {
        0u32
    } else {
        b.emit(DevOp::Embed, rows.clone(), &[], |d| {
            d.t[0] = n.x;
            d.t[1] = n.emb;
            d.t[2] = n.ids;
            d.i[0] = t;
            d.i[1] = c.hidden;
            d.f[0] = escale;
        })
    };

    // In decode, every projection is a GEMV (M=1): a 32x32 matrix core would run with 1 of
    // 32 M-lanes live, and the step is bandwidth-bound on the 57 GiB of weights anyway.
    // In DECODE the RMSNorm is folded into the consuming GEMV (norm mode 2: the GEMV computes
    // the row RMS itself, from the x it already staged in LDS). That deletes the RMSNORM
    // packet, its gate, AND its single-CU serialisation -- a decode norm is a row reduction,
    // so exactly ONE workgroup could do it while the other 255 waited on the counter.
    //
    // In PREFILL the norm stays its own packet: it has T rows, so it already parallelises, and
    // folding it into the GEMM (GEMM_NORM) is a measured LOSS -- the A tile is re-fetched once
    // per N-tile, so the per-element norm work gets multiplied by N/BN.
    let eps = c.eps;
    // `w8`/`scale` are the fp8 twin of the bf16 weight `w` and its per-channel dequant scale; they
    // are used ONLY on the decode fp8 path (DevOp::GemvFp8). Prefill and bf16 decode ignore them.
    // `xq`/`ascale_t` are the T8 w8a8 fp8-quantized activation twin of `a` and its per-row a_scale;
    // they are TENSOR_NONE (and ignored) off the w8a8 path. On the w8a8 path the caller has already
    // emitted the shared QuantFp8 (once per activation site) and threaded its id into `deps`.
    let proj = |b: &mut Builder,
                out: u32,
                a: u32,
                w: u32,
                w8: u32,
                scale: u32,
                xq: u32,
                ascale_t: u32,
                m: u32,
                nn: u32,
                k: u32,
                gamma: u32,
                cus: Vec<u32>,
                deps: &[u32]|
     -> u32 {
        if gemv_family && fp8 {
            return b.emit(DevOp::GemvFp8, cus, deps, |d| {
                d.t[0] = out;
                d.t[1] = a;
                d.t[2] = w8;
                d.t[5] = scale;
                d.i[0] = m;
                d.i[1] = nn;
                d.i[2] = k;
                d.i[4] = 0;
            });
        }
        // PREFILL fp8 tiled GEMM. Two builds share the GEMM_FP8 opcodes; the interp cubin picks the
        // kernel by PLOW_NV_W8A8. T6 w8a16 (default cubin): bf16 activation (t1=a), e4m3 weight (t2)
        // + per-channel dequant scale (t4). T8 w8a8 (PLOW_NV_W8A8 cubin, PLOW_W8A8 emit): BOTH
        // operands e4m3 — t1=xq (per-row-quantized activation), t3=a_scale, t2=w8, t4=w_scale — true
        // mma.sync.m16n8k32. The opcode tracks whatever tile pick_tile would have chosen for bf16.
        if !gemv_family && fp8 {
            let op = match pick_tile(m, nn, k, n_cu) {
                DevOp::GemmMed => DevOp::GemmMedFp8,
                DevOp::GemmSmall => DevOp::GemmSmallFp8,
                _ => DevOp::GemmFp8,
            };
            return b.emit(op, cus, deps, |d| {
                d.t[0] = out;
                d.t[2] = w8;
                d.t[4] = scale;
                if w8a8 {
                    d.t[1] = xq;
                    d.t[3] = ascale_t;
                } else {
                    d.t[1] = a;
                }
                d.i[0] = m;
                d.i[1] = nn;
                d.i[2] = k;
                d.i[4] = 0;
            });
        }
        let fold = gemv_family && gamma != TENSOR_NONE;
        let op = if gemv_family {
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

    // T8 w8a8: emit the ONE shared per-row fp8 activation quant (DevOp::QuantFp8) that a group of
    // GEMMs reading the same activation depends on — the linchpin of correctness. A per-proj quant
    // would race (q's quant would overwrite the xq that k/v read); the single shared quant is
    // required, not merely an optimization. `after` is the producer of `src` (the norm/attn output);
    // the returned id is what the consuming GEMMs must wait on. Off the w8a8 path it is inert and
    // returns `after`, so every caller can thread it uniformly. Row-sliced across `rows` blocks.
    let quant = |b: &mut Builder, xq: u32, ascale_t: u32, src: u32, k: u32, after: u32| -> u32 {
        if !w8a8 {
            return after;
        }
        b.emit(DevOp::QuantFp8, rows.clone(), &[after], |d| {
            d.t[0] = xq;
            d.t[1] = src;
            d.t[2] = ascale_t;
            d.i[0] = t;
            d.i[1] = k;
        })
    };

    // Qwen/Llama PRE-NORM decode fuses each (residual add, RMSNorm) pair into ONE AddNorm packet
    // (see the AddNorm emits in the loop). Deletes 72 packets/token and, more importantly, 72
    // global gates off the critical path — decode here is fixed per-gate tax, not weight streaming.
    let fuse_norm = c.arch != Arch::Gemma4 && gemv_family;
    // Gemma SANDWICH decode fuses each (NormResidual, following RMSNorm) pair into ONE
    // NormResidualNorm packet (Experiment N1) — the narrow→narrow successor to AddNorm. Same two
    // sites as fuse_norm (post-attn→pre-ffn, and end-of-layer→next input norm), but the residual is
    // a post-normed sandwich add with a per-layer scale, not a plain sum. Deletes a gate + an HBM
    // round trip per fused pair. Decode only; prefill keeps the split (T rows parallelise the norm).
    let gfuse = c.arch == Arch::Gemma4 && gemv_family;

    for l in block.clone() {
        let full = c.is_full[l];
        // MIXED fp8-KV (PLOW_FP8_KV_FULL=1): per-layer effective flag — see declare(). Ops keyed
        // on it (HeadNormRope[Fp8], FlashDecode[Fp8], FlashPrefill[Fp8], the fp8-tuned nsplit
        // gates) all follow the LAYER's cache dtype.
        let fp8_kv = fp8_kv && (full || !fp8_kv_full);
        let hd = if full { c.hd_full } else { c.hd_slide };
        // this rank's kv-heads, with shared-kv-head replication for tp > kvh (§3a/§13.2, kvh_local).
        let kvh = kvh_local(if full { c.kvh_full } else { c.kvh_slide }, tp, l as u32);
        let qd = heads * hd; // column-parallel q output shard
        let kd = kvh * hd; // column-parallel k/v output shard
        let (cs, sn) = if full {
            (n.cos_f, n.sin_f)
        } else {
            (n.cos_s, n.sin_s)
        };
        let win = if full { 0 } else { c.window };
        let w = &n.lw[l];
        // k_eq_v is Gemma-full-layer only; Llama/Qwen always have a real v_proj even though every
        // layer is "full". skip_norm bypasses the RMS in HeadNormRope: Llama has no q/k norm and
        // neither model norms V.
        let keqv = full && c.k_eq_v;
        let qk_skip: u32 = if c.has_qk_norm { 0 } else { 1 };
        let v_skip: u32 = if c.has_v_norm { 0 } else { 1 };
        // FUSED Q|K|V (decode, real v_proj): one GEMV packet computes all three projections, on
        // all CUs. Two fewer gates than split3 AND uniform fill instead of the 171/42/43 CU split.
        // Gemma's k_eq_v layers keep the old path (no v_proj to fuse). See DevOp::GemvQkv.
        // FP8 has no QKV-fusion arm (opcode 26 deferred): q/k/v run as three separate GEMV_FP8.
        // T11 packet-reduction probe: PLOW_NO_FUSE_QKV=1 reverts to the historical split3 path
        // (q/k/v as three separate bf16 Gemv packets = +2 packets/layer, uneven CU fill). Tokens
        // are bit-identical (each output column is the same per-column dot). Off by default =>
        // byte-identical stream. Measures the marginal TPOT cost of a 2-gate/layer reduction.
        let fuse_qkv = gemv_family
            && !keqv
            && !fp8
            && std::env::var("PLOW_NO_FUSE_QKV").ok().as_deref() != Some("1");

        // GQA FUSION changes the decode split, and the two have to agree or the machine idles.
        //
        // A fused flash-decode work item is (kv_head, split), not (query_head, split) — it reads
        // each KV row ONCE and dots it against all GF query heads sharing it. That divides
        // n_work by GF, so `nsplit` must be multiplied by it to keep 256 work units on 256 CUs.
        // The kernel picks GF from head_dim (PLOW_FA_GF in dev_isa.h); here we derive nsplit from
        // kv_heads, which is the same statement from the other side.
        //
        // It is PER LAYER because kv_heads is: 16 on a sliding layer (GQA 2), 4 on a full one
        // (GQA 8). A single nsplit for both would leave the full layers on 4 of 256 CUs.
        // The sliding layers' cache is a RING; the full layers' is linear. `kvm` is 0xFFFFFFFF
        // for a full layer, so the AND in the kernels is a no-op there. See kv_rows().
        let (kvr, kvm) = kv_ring(full, ctx);
        // GF is the flash-decode GQA fusion factor: query heads carried by ONE work item, and it is
        // the KERNEL constant PLOW_FA_GF(hd) = PLOW_FA_GF_FULL (default 2) — NOT 8. The compiler and
        // kernel must agree (dev_isa.h). GF=2 fuses sliding layers fully (GQA 2) and full layers
        // partially (GQA 8 -> reads each row 4x). Under tp=8 shared-kv-head replication a full layer
        // is GQA 4 locally, still a clean multiple of GF=2. The binding invariant is gqa_local % GF.
        let gf = FA_GF_FULL; // MUST track the kernel's PLOW_NV_FA_GF / PLOW_FA_GF_FULL
        assert_eq!(
            (heads / kvh) % gf,
            0,
            "layer {l}: GF {gf} must divide GQA {}",
            heads / kvh
        );
        // n_work = n_head/GF * nsplit. Filling all 256 CUs would want nsplit = n_cu*GF/n_head
        // (= 64 on a full layer), and that is WRONG in both directions: it fragments flash into
        // 52-row work items whose per-item overhead swamps them, and it multiplies flash_merge's
        // partials by GF (merge is only 32 workgroups, so it scales with nsplit).
        //
        // We do not need to refill the machine: the fusion cut the traffic by GF, so fewer CUs
        // each doing GF-times-less work can still finish sooner. Swept on the real model:
        //
        //     nsplit   8     16     32     64
        //     token   16.8  16.8   17.8   19.3   ms
        //
        // 16 it is. Above that, flash fragments and merge (only 32 workgroups, and it scales
        // with nsplit) takes back everything the fusion won.
        // CONTEXT-ADAPTIVE default (short-ctx-flash lever). At SHORT ctx the KV is small, so ns16
        // OVER-splits: flash_merge's crit-path busy scales with nsplit (MEASURED Gemma-4-31B TP=1
        // ctx1k: merge busy 1010us@ns16 vs 764us@ns8) and the Opart f32 partials scale with it too,
        // while flash_decode barely benefits from >8 splits when there is little KV to read. At LONG
        // ctx the big full-layer KV read DOES need the fuller split. MEASURED decode ms/tok (ns8/ns16):
        //   ctx    1k          8k           64k
        //   ns8    18.06       18.82        24.67
        //   ns16   18.34       18.82        22.01
        // ns8 wins <=8k (-0.28ms @1k), ties at 8k, and REGRESSES >8k — so gate on the pkt's max_ctx.
        // ns1/ns2 lose EVERYWHERE (flash_decode serialization: busy 3541us@1k, 45189us@64k) — the
        // merge-elision path is a dead end: split-KV PARALLELISM, not the merge, is the ceiling.
        // Default mul: 1 (ns8) for a short-ctx pkt, 2 (ns16) otherwise. PLOW_NS_MUL / PLOW_NS_ABS
        // still override. Crossover measured at ~8k; a pkt compiled for <=8k is a short-ctx pkt.
        let mul_default: u32 = if ctx <= 8192 { 1 } else { 2 };
        let mul: u32 = std::env::var("PLOW_NS_MUL")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(mul_default);
        // DECODE nsplit fill target uses the UNSHARDED head count (c.heads), NOT this rank's
        // sharded `heads` (= c.heads/tp). Under Megatron TP the KV cache is HEAD-partitioned (each
        // rank owns c.heads/tp q-heads over the FULL ctx per head), not context-split — so the right
        // per-head split of the KV context is tp-INDEPENDENT. The old `div_ceil(heads)` inflated
        // nsplit by tp (16->64 at tp=4) to refill 256 CUs, which quadrupled flash_merge: merge runs
        // on only heads/tp workgroups, each then reducing tp× the partials — MEASURED 1.00->3.05 ms
        // of a 14.55 ms tp=4 token. Basing the fill on c.heads keeps nsplit=16 at every tp (merge
        // per head unchanged), recovering ~2.3 ms at tp=4 (14.55->12.2). tp==1: c.heads==heads, so
        // byte-identical to the pre-TP path. Decode fragments flash_decode's fill under TP (fewer
        // work-items than CUs) but per-rank flash work is 1/tp anyway; merge, on the crit path, wins.
        let ns = if gemv_family {
            (n_cu * mul).div_ceil(c.heads).max(1)
        } else {
            ns
        };
        // sm_120 (188 SMs) 31B FULL-LAYER OVERSPLIT (campaign T7-31b-decode, RTX PRO 6000). The
        // CU-fill formula gives ns=12 on the 188-SM card at 32 heads (188*2/32), a 1.0x flash fill
        // (n_grp*ns = 16*12 = 192 ≈ 188 SMs). MEASURED: ns16 (1.36x oversubscribe) beats ns12 at
        // EVERY ctx for the dense 31B — the fine full-layer KV splits (10 layers × kv4 × hd512) hide
        // the long-ctx read latency and the 32-workgroup merge still absorbs the extra partials:
        //   ctx      1k      4k      16k     32k     64k     128k    (bf16 ms/tok)
        //   ns12     47.58   47.95   49.58   51.84   55.93   64.11
        //   ns16     47.49   47.79   48.98   50.73   53.95   60.37   (-0.2% .. -5.8%)
        //   ns24     -       -       -       51.16   -       61.59   (over-split: worse than 16)
        // fp8 (weight-only, so identical KV/flash bytes) gains MORE at long ctx: 64k -5.6%, 128k too.
        // Gated to the 31B signature — mixed sliding/full attention with 4-KV full layers — so 12B
        // (kvh_full=1 → ns24 already), Qwen/Llama (kvh_full==kvh_slide, want ns4-8), and short-ctx
        // pkts (<=8192, untested here) are byte-identical. PLOW_NS_MUL/PLOW_NS_ABS still override.
        let ns = if gemv_family && ctx > 8192 && c.kvh_full >= 4 && c.kvh_slide != c.kvh_full {
            ns.max(16)
        } else {
            ns
        };
        // GRID-ALIGNED FULL-LAYER nsplit (T9b-31b-tune, RTX PRO 6000 / 188 SMs). The full
        // layers' flash-decode work is n_grp*nsplit = (heads/GF)*nsplit items spread over n_cu
        // resident blocks. With n_grp=16 and n_cu=188 (gcd 4) that count is RAGGED at every
        // nsplit that is not a multiple of n_cu/gcd = 47: ceil() leaves ~68 blocks doing 2x the
        // work while the rest do 1x, and FLASH_MERGE waits for the slow 2x blocks (MEASURED
        // block-0 gate 658k cyc/op @128k, T9b trace). Rounding the fill target UP to a multiple
        // of `aligned` makes every block do exactly the same number of items.
        //   MEASURED 31B decode @128k (method of record, 120 timed): ns16(base) 58.57 ->
        //   ns47(aligned) 56.60 ms = -3.4%. ns24 (=384 items, ceil 3/block, WORSE imbalance)
        //   was 59.59, SLOWER than base — proving ALIGNMENT, not split count, is the lever
        //   (H2 stopped at ns24 and missed this). Only the 10 hd512/kv4 FULL layers change;
        //   the 50 hd256 sliding layers keep ns16 (their window-1024 KV is tiny, so 47-way
        //   over-splitting them would only add merge partials). Gated to the same 31B long-ctx
        //   signature as the ns.max(16) floor above, plus `full`, plus a <=64 sanity cap so a
        //   shape whose n_grp is coprime to n_cu (aligned would jump to n_cu) falls back.
        //   PLOW_NS_FULL_ABS still overrides for sweeps.
        let ns = if gemv_family && full && ctx > 8192 && c.kvh_full >= 4 && c.kvh_slide != c.kvh_full
        {
            let n_grp = (heads / FA_GF_FULL).max(1);
            let aligned = n_cu / gcd(n_grp, n_cu); // smallest grid-aligned nsplit step
            let cand = ns.div_ceil(aligned) * aligned; // round the ns16 target up to it
            if cand <= 64 {
                cand
            } else {
                ns
            }
        } else {
            ns
        };
        // GRID-ALIGNED FULL-LAYER nsplit, 12B SINGLE-GLOBAL-KV-HEAD signature (beat12b-fp8-margin).
        // Gemma-4-12B full layers are kvh_full=1 (ONE kv head serves all 16 q heads): the CU-fill
        // formula gives ns=24 -> n_grp(8)*24 = 192 items on 188 SMs — RAGGED: 4 blocks run 2 items,
        // 184 run 1, and FLASH_MERGE waits for the 2x stragglers, so the full-layer flash runs at
        // ~2x its aligned latency at long ctx. Rounding to a multiple of n_cu/gcd(n_grp,n_cu)=47
        // (376 items = exactly 2/block) fixes it. MEASURED (flashdec_fp8_bw_12b microbench +
        // gemma4_sm120_chat, fp8 weights + fp8 head + fp8-KV, method of record n=112):
        //   decode ms/tok @128k: ns24 16.163 -> ns47 13.988 (-13.5%);  @1k 11.219 -> 11.213 (free)
        // Gated on fp8_kv (an emit-time flag) because the bf16-KV optimum differs (bf16 @128k
        // prefers ns23; microbench 0.436 vs 0.497) — with PLOW_FP8_KV unset the packet stays
        // byte-identical. Same <=128 sanity cap idea as the 31B block; PLOW_NS_ABS/PLOW_NS_FULL_ABS
        // still override below.
        let ns = if gemv_family && full && ctx > 8192 && c.kvh_full == 1 && fp8_kv {
            let n_grp = (heads / FA_GF_FULL).max(1);
            let aligned = n_cu / gcd(n_grp, n_cu);
            let cand = ns.div_ceil(aligned) * aligned;
            if cand <= 128 {
                cand
            } else {
                ns
            }
        } else {
            ns
        };
        // WINDOWED-LAYER nsplit cap (beat12b-fp8-margin). A sliding layer's flash span never
        // exceeds `win` rows, so the CU-fill ns=24 over-splits it into 43-row items AND lands on
        // the same ragged 192-items-on-188-SMs grid as the full layers — a FIXED per-token cost
        // (the window doesn't grow with ctx). Cap ns so an item keeps >= 64 rows (a quarter
        // FA_DEC_TILE): win=1024 -> ns 16, n_work = 8*16 = 128 <= 188, no 2x tail. MEASURED
        // (12B fp8kv decode, sliding ns sweep at full ns=47, ms/tok @1k):
        //   ns8 10.937 | ns12 10.956 | ns16 10.921 | ns23 10.990 | ns24 (base) 11.221 | ns47 11.212
        // -0.30 ms at EVERY ctx (@128k 13.978 -> 13.684). fp8_kv-gated like the block above so
        // flag-unset packets stay byte-identical; PLOW_NS_ABS still overrides below.
        let ns = if gemv_family && !full && win > 0 && fp8_kv {
            ns.min((win / 64).max(1))
        } else {
            ns
        };
        // DECODE nsplit ABSOLUTE OVERRIDE (occupancy tuning). PLOW_NS_MUL scales the CU-fill target;
        // PLOW_NS_ABS pins nsplit directly. MEASURED on Qwen3-4B (all-global, GQA 4, MI350X):
        // the default mul=2 (ns=16) OVER-SPLITS flash_decode — each split's fixed overhead (Q
        // re-staging + the flash_merge partial + its barriers) dominates the tiny per-split KV
        // work, so summed flash_decode work grows with nsplit (ns 4/8/16/32/64 -> 59/80/112/174/285
        // ms) and decode ms/tok is best at ns=4-8 (4.3-4.4) vs 4.6 at ns=16. flash_decode is
        // over-fragmented, not under-filled. Inert by default; leaves Gemma's tuned mul path alone.
        let ns = std::env::var("PLOW_NS_ABS")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|_| gemv_family)
            .unwrap_or(ns);
        // Full-attention-only decode split override. Unlike PLOW_NS_ABS this does not also
        // over-split Gemma's many hd256 sliding layers. It is the controlled sweep knob for
        // full-layer GQA-fusion experiments on sm_120 (GF4/ns24 => 8 groups * 24 = 192 work
        // items on the 188-SM RTX PRO 6000). Default unset preserves every existing packet.
        let ns = std::env::var("PLOW_NS_FULL_ABS")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|_| gemv_family && full)
            .unwrap_or(ns);

        // The norm is ONE packet whose result all of q/k/v share.
        //
        // Folding it into each GEMV instead (norm mode 2, where the GEMV recomputes the row RMS
        // from its LDS-staged x) is CORRECT and deletes the packet and its gate -- and it was
        // MEASURED SLOWER: 22.4 -> 24.4 ms/token. Five consumers (q, k, v, gate, up) then each
        // redo the reduction, so one shared 10 us norm becomes five redundant ones on the
        // critical path, and the two gates saved do not pay for it. The op still supports mode
        // 2 (it is right for a single consumer), but the compiler does not use it here.
        // The end-of-layer AddNorm ALSO produces the NEXT layer's normed input, so for l>0 the
        // input RMSNorm is already done and `dep` carries n.hn directly.
        let c_n = if (fuse_norm || gfuse) && l > block.start {
            dep // previous layer's end-of-layer fused norm already wrote the normed n.hn
        } else {
            // The block's FIRST layer reads the uploaded `act.x` with no producer
            // (Embed was skipped), so its RmsNorm depends on `&[]`. Full model =>
            // block.start==0 and this only affects l==0, whose dep IS the Embed.
            let nd: &[u32] = if block_mode && l == block.start { &[] } else { &[dep] };
            b.emit(DevOp::RmsNorm, rows.clone(), nd, |d| {
                d.t[0] = n.hn;
                d.t[1] = n.x;
                d.t[2] = w.g_in;
                d.i[0] = t;
                d.i[1] = c.hidden;
                d.f[0] = c.eps;
            })
        };
        let (qkv_src, qkv_g) = (n.hn, TENSOR_NONE);

        // q, k and v are INDEPENDENT. Running all three on all 256 CUs makes them serialise
        // behind three separate counter gates for no reason: they are bandwidth-bound, so
        // total time is (total weight bytes / aggregate bandwidth) either way -- but disjoint
        // CU sets put them in flight together and cost ONE gate instead of three. Split in
        // proportion to weight bytes so they finish together.
        let (nq, nk, nv);
        let (c_q, c_k, c_v, v_src);
        if fuse_qkv {
            // ONE packet on all CUs: cols [0,qd) -> q, [qd,qd+kd) -> k, [qd+kd,qd+2kd) -> v.
            (nq, nk, nv) = (n_cu, n_cu, n_cu); // unused: fused headnorm deps are coarse
            let fused = b.emit(DevOp::GemvQkv, all.clone(), &[c_n], |d| {
                d.t[0] = n.qg;
                d.t[1] = qkv_src;
                d.t[2] = w.wq;
                d.t[3] = n.kg;
                d.t[4] = w.wk;
                d.t[5] = n.vg;
                d.t[6] = w.wv;
                d.i[0] = t;
                d.i[1] = qd;
                d.i[2] = c.hidden;
                d.i[3] = kd;
                d.i[4] = kd;
            });
            (c_q, c_k, c_v, v_src) = (fused, fused, fused, n.vg);
            let _ = qkv_g; // norm is a shared packet here, never folded into the fused GEMV
        } else {
            let (cq, ck, cv) = if gemv_family {
                split3(n_cu, qd, kd, if keqv { 0 } else { kd })
            } else {
                split3(
                    n_cu,
                    tiles(t, qd),
                    tiles(t, kd),
                    if keqv { 0 } else { tiles(t, kd) },
                )
            };
            (nq, nk, nv) = (cq.len() as u32, ck.len() as u32, cv.len() as u32);
            // w8a8: ONE quant of the (hidden-width) attn input, shared by q/k/v.
            let dq = quant(b, n.xqh, n.ash, qkv_src, c.hidden, c_n);
            let cqc = proj(
                b,
                n.qg,
                qkv_src,
                w.wq,
                w.wq8,
                w.sq,
                n.xqh,
                n.ash,
                t,
                qd,
                c.hidden,
                qkv_g,
                cq,
                &[dq],
            );
            let ckc = proj(
                b,
                n.kg,
                qkv_src,
                w.wk,
                w.wk8,
                w.sk,
                n.xqh,
                n.ash,
                t,
                kd,
                c.hidden,
                qkv_g,
                ck,
                &[dq],
            );
            let (vsrc, cvc) = if keqv {
                (n.kg, ckc) // k_eq_v: V is the RAW k_proj output
            } else {
                (
                    n.vg,
                    proj(
                        b,
                        n.vg,
                        qkv_src,
                        w.wv,
                        w.wv8,
                        w.sv,
                        n.xqh,
                        n.ash,
                        t,
                        kd,
                        c.hidden,
                        qkv_g,
                        cv,
                        &[dq],
                    ),
                )
            };
            (c_q, c_k, c_v, v_src) = (cqc, ckc, cvc, vsrc);
        }

        // headnorm+RoPE for q; and for k/v the store goes STRAIGHT INTO THE KV CACHE at
        // out_row0. In decode that row is the current position, which the runtime patches.
        let hn_cus: Vec<u32> = (0..((t * heads).div_ceil(8)).min(n_cu).max(1)).collect();
        let nhn = hn_cus.len() as u32;

        // gemv -> headnorm. headnorm workgroup j owns whole HEADS, and head h is the 256 (or
        // 512) consecutive output columns [h*hd, h*hd+hd) of the projection — so it needs only
        // the handful of gemv workgroups that produced those columns, not all 128.
        //
        // This is ONLY sparse under GV_BLOCKED (op_gemm.h). With the default wave-interleaved
        // column assignment a gemv workgroup's columns are scattered across all of N, and the
        // fan-in is 128 of 128 — measured, and the reason the first attempt at a fine chain
        // bought nothing.
        let hn_dep = |gemv: u32, nblk_g: u32, nheads: u32| -> Vec<Dep> {
            if !gemv_family || fuse_qkv {
                // the gemv column map assumes d_gemv (GV_BLOCKED); prefill is
                // a GEMM. The fused q|k|v op concatenates all three projections' columns across the
                // SAME 256 workgroups, so a head's per-workgroup producer set is no longer the
                // single-projection map below — fall back to coarse (the fused op is one uniform
                // packet, so all workgroups finish together and coarse costs ~nothing).
                // NOTE: we DECLARE the fine edge; `select_granularity` decides if it survives.
                return vec![Dep::Coarse(gemv)];
            }
            let dim = nheads * hd; // the projection's N
            let map = (0..nhn)
                .map(|j| {
                    let mut s: Vec<u32> = headnorm_items(nhn, t * nheads, j)
                        .into_iter()
                        .flat_map(|w| {
                            let h = w % nheads; // item = token*nheads + head
                            gemv_wgs_for_cols(dim, nblk_g, h * hd, (h + 1) * hd)
                        })
                        .collect();
                    s.sort_unstable();
                    s.dedup();
                    s
                })
                .collect();
            vec![Dep::Fine {
                producer: gemv,
                map,
            }]
        };

        let c_qn = b.emit_dep(
            DevOp::HeadNormRope,
            hn_cus.clone(),
            hn_dep(c_q, nq, heads),
            |d| {
                d.t[0] = n.q;
                d.t[1] = n.qg;
                d.t[2] = w.qn;
                d.t[3] = cs;
                d.t[4] = sn;
                d.t[5] = n.pos;
                d.i[0] = t;
                d.i[1] = heads;
                d.i[2] = hd;
                d.i[3] = 0;
                d.i[4] = qk_skip;
                d.f[0] = c.eps;
            },
        );
        // fp8-KV: the k/v norm STORES the cache as e4m3 with a per-row scale (t6). q is unchanged
        // (it is not cached — flash reads it as bf16), so it stays plain HeadNormRope.
        let hn_op = if fp8_kv {
            DevOp::HeadNormRopeFp8
        } else {
            DevOp::HeadNormRope
        };
        let c_kn = b.emit_dep(hn_op, hn_cus.clone(), hn_dep(c_k, nk, kvh), |d| {
            d.t[0] = n.kc[l];
            d.t[1] = n.kg;
            d.t[2] = w.kn;
            d.t[3] = cs;
            d.t[4] = sn;
            d.t[5] = n.pos;
            d.t[6] = n.kcs[l]; // fp8-KV per-row scale (NONE in bf16 mode)
            d.i[0] = t;
            d.i[1] = kvh;
            d.i[2] = hd;
            d.i[3] = 0;
            d.i[4] = qk_skip;
            d.f[0] = c.eps;
            // j0 = the KV cache's row stride (the RING size on a sliding layer); j1 = the row
            // mask. The write lands in the HEAD-MAJOR cache so flash can stream a head
            // end-to-end. See PLOW_KV_RING in dev_isa.h.
            d.j[0] = kvr;
            d.j[1] = kvm;
            // BATCH>1 decode: i6 = n_batch_kv selects the per-sequence KV ring (each seq writes at
            // its own pos[t]). 0 for prefill/B=1 => legacy single-ring, byte-identical.
            if decode && t > 1 {
                d.i[6] = t;
            }
        });
        if decode {
            kv_rows.push(c_kn);
        }
        // v_norm: WEIGHTLESS (gamma NONE) and NO RoPE (cos NONE).
        // On a full layer V comes from the RAW k_proj output, so its producer is c_k (nk wgs).
        let vn_dep = if keqv {
            hn_dep(c_v, nk, kvh)
        } else {
            hn_dep(c_v, nv, kvh)
        };
        let c_vn = b.emit_dep(hn_op, hn_cus, vn_dep, |d| {
            d.t[0] = n.vc[l];
            d.t[1] = v_src;
            d.t[5] = n.pos;
            d.t[6] = n.vcs[l]; // fp8-KV per-row scale (NONE in bf16 mode)
            d.i[0] = t;
            d.i[1] = kvh;
            d.i[2] = hd;
            d.i[3] = 0;
            d.i[4] = v_skip;
            d.f[0] = c.eps;
            d.j[0] = kvr;
            d.j[1] = kvm;
            if decode && t > 1 {
                d.i[6] = t;
            }
        });
        if decode {
            kv_rows.push(c_vn);
        }

        // headnorm -> flash. A flash work item is (batch, head, split); it reads Q for its own
        // head and the KV cache for head/gqa. Every OTHER row of the cache was written by a
        // PREVIOUS decode step (a previous launch), so within this program flash depends only on
        // the three headnorms' work for its own head — not on all of them.
        let fa_dep = || -> Vec<Dep> {
            if !gemv_family {
                return vec![Dep::Coarse(c_qn), Dep::Coarse(c_kn), Dep::Coarse(c_vn)];
            }
            let nblk_f = all.len() as u32;
            let gqa = heads / kvh;
            let n_work = t * heads * ns; // d_flash_decode: n_batch * n_head * nsplit
            let mk = |kv: bool| -> Vec<Vec<u32>> {
                (0..nblk_f)
                    .map(|f| {
                        let mut s: Vec<u32> = (0..n_work)
                            .filter(|w| w % nblk_f == f) // the items THIS workgroup runs
                            .map(|w| {
                                let h = (w / ns) % heads;
                                let bb = w / (ns * heads);
                                if kv {
                                    headnorm_wg_of(nhn, bb * kvh + h / gqa)
                                } else {
                                    headnorm_wg_of(nhn, bb * heads + h)
                                }
                            })
                            .collect();
                        s.sort_unstable();
                        s.dedup();
                        s
                    })
                    .collect()
            };
            vec![
                Dep::Fine {
                    producer: c_qn,
                    map: mk(false),
                },
                Dep::Fine {
                    producer: c_kn,
                    map: mk(true),
                },
                Dep::Fine {
                    producer: c_vn,
                    map: mk(true),
                },
            ]
        };
        let c_fa = if gemv_family {
            let fa_op = if fp8_kv {
                DevOp::FlashDecodeFp8
            } else {
                DevOp::FlashDecode
            };
            b.emit_dep(fa_op, all.clone(), fa_dep(), |d| {
                d.t[0] = n.opart;
                d.t[1] = n.mlpart;
                d.t[2] = n.q;
                d.t[3] = n.kc[l];
                d.t[4] = n.vc[l];
                d.t[5] = n.kvlen;
                d.t[6] = n.kcs[l];
                d.t[7] = n.vcs[l]; // fp8-KV per-row scales (NONE in bf16 mode)
                                   // BATCH>1: n_batch = t (one query row per sequence, each with its own KV ring).
                d.i[0] = t;
                d.i[1] = heads;
                d.i[2] = kvh;
                d.i[3] = kvr;
                d.i[4] = win;
                d.i[5] = ns;
                d.i[6] = hd;
                d.i[7] = kvm;
                d.f[0] = c.attn_scale;
                // KV row-capacity for the PLOW_NV_KVBOUNDS trap: the b>=1 OOB read past an
                // under-sized KV allocation traps here instead of reading fluent wrong text. Set
                // only for B>1 so the B=1 packet stays byte-identical (j[0] default 0 => no check).
                if t > 1 {
                    d.j[0] = t * kvh * kvr;
                }
            })
        } else {
            let fa_op = if fp8_kv {
                DevOp::FlashPrefillFp8
            } else {
                DevOp::FlashPrefill
            };
            b.emit(fa_op, all.clone(), &[c_qn, c_kn, c_vn], |d| {
                d.t[0] = n.opart;
                d.t[1] = n.mlpart;
                d.t[2] = n.q;
                d.t[3] = n.kc[l];
                d.t[4] = n.vc[l];
                d.t[6] = n.kcs[l];
                d.t[7] = n.vcs[l]; // fp8-KV per-row scales (NONE in bf16 mode)
                                   // Fused epilogue: t[5] is the final bf16 attention output (n.at). When !fused
                                   // it stays NONE and flash_prefill writes the f32 partial for d_flash_merge.
                if fused {
                    d.t[5] = n.at;
                }
                // n_head is THIS RANK's sharded head count under TP (§3a): Q/O buffers hold
                // heads=c.heads/tp heads and flash_merge reads the same `heads`. Passing the
                // unsharded c.heads here read past the sharded Q buffer (the prefill-TP bug).
                d.i[0] = t;
                d.i[1] = t;
                d.i[2] = heads;
                d.i[3] = kvh;
                d.i[4] = 0;
                d.i[5] = win;
                d.i[6] = hd;
                d.i[7] = ns;
                d.f[0] = c.attn_scale;
                d.j[0] = kvr;
                d.j[1] = kvm; // head-major; RING on a sliding layer
            })
        };
        // When fused, flash_prefill already wrote the normalized bf16 to n.at, so there is no
        // FlashMerge op and o_proj depends on the flash op directly. Coarse: n.at row r needs
        // every head of its q-tile, which is spread across the flash workgroups.
        let attn_dep = if fused {
            c_fa
        } else {
            let mg_cus: Vec<u32> = (0..(t * heads).min(n_cu).max(1)).collect();
            let fill = |d: &mut DevInst| {
                d.t[0] = n.at;
                d.t[1] = n.opart;
                d.t[2] = n.mlpart;
                d.i[0] = t;
                d.i[1] = heads;
                d.i[2] = ns;
                d.i[3] = hd;
            };
            // A merge workgroup folds the `ns` partials of its own (row, head) and reads
            // nothing else, so make it wait for exactly those flash slices instead of all 256.
            let map = flash_merge_map(
                t * heads,
                ns,
                if gemv_family { 1 } else { Q_TILE_ROWS },
                heads,
                all.len() as u32,
                mg_cus.len() as u32,
            );
            b.emit_dep(
                DevOp::FlashMerge,
                mg_cus,
                vec![Dep::Fine {
                    producer: c_fa,
                    map,
                }],
                fill,
            )
        };

        // o_proj is ROW-parallel (plans/tp-design.md §3a): input = this rank's qd heads, output =
        // the FULL H-vector but only a PARTIAL sum (this rank's head contribution). Under TP it
        // writes that partial into the peer-mapped og_tp slot and an XReduce sums the N peers'
        // partials into the replicated `og` that NormResidual consumes — all-reduce #1 of the layer.
        // proj() picks the fp8 (GemvFp8) arm on the decode fp8 path via the wo8/so operands.
        // w8a8: quant the (qd-width) attention output feeding o_proj.
        let do_ = quant(b, n.xqo, n.aso, n.at, qd, attn_dep);
        let c_o = if tp > 1 {
            let c_op = proj(
                b,
                n.og_tp,
                n.at,
                w.wo,
                w.wo8,
                w.so,
                n.xqo,
                n.aso,
                t,
                c.hidden,
                qd,
                TENSOR_NONE,
                all.clone(),
                &[do_],
            );
            emit_xreduce(b, &mut xgate, decode, &xr_cus, c_op, n.og, xr_elems, tp, 0)
        } else {
            proj(
                b,
                n.og,
                n.at,
                w.wo,
                w.wo8,
                w.so,
                n.xqo,
                n.aso,
                t,
                c.hidden,
                qd,
                TENSOR_NONE,
                all.clone(),
                &[do_],
            )
        };
        // FIRST RESIDUAL + PRE-MLP NORM — the biggest structural fork.
        //   Gemma SANDWICH: x = x + post_attn_norm(o); then hn = pre_feedforward_norm(x).
        //   Llama/Qwen PRE-NORM: x = x + o (plain); then hn = post_attention_layernorm(x).
        // Gemma applies its post-attn norm to the ATTENTION OUTPUT before the add; Llama/Qwen add
        // the raw output and normalize the residual stream going INTO the MLP.
        let gemma = c.arch == Arch::Gemma4;
        // Pre-MLP norm. Gemma: sandwich (NormResidual) then a separate pre-FF norm. Qwen/Llama
        // decode: x += o, then post_attention_layernorm(x) — fused into ONE AddNorm. Qwen/Llama
        // prefill keeps the split (T rows already parallelise the norm; a parallel agent owns it).
        let c_pf = if fuse_norm {
            b.emit(DevOp::AddNorm, rows.clone(), &[c_o], |d| {
                d.t[0] = n.hn;
                d.t[1] = n.x;
                d.t[2] = n.x;
                d.t[3] = n.og;
                d.t[4] = w.g_pa;
                d.i[0] = t;
                d.i[1] = c.hidden;
                d.f[0] = c.eps;
            })
        } else if gfuse {
            // x = x + post_attn_norm(o); hn = pre_feedforward_norm(x) — Gemma sandwich in ONE packet.
            b.emit(DevOp::NormResidualNorm, rows.clone(), &[c_o], |d| {
                d.t[0] = n.hn;
                d.t[1] = n.x;
                d.t[2] = n.x;
                d.t[3] = n.og;
                d.t[4] = w.g_pa;
                d.t[5] = w.g_pf;
                d.i[0] = t;
                d.i[1] = c.hidden;
                d.f[0] = c.eps;
                d.f[1] = 1.0;
            })
        } else {
            let c_r1 = if gemma {
                b.emit(DevOp::NormResidual, rows.clone(), &[c_o], |d| {
                    d.t[0] = n.x;
                    d.t[1] = n.x;
                    d.t[2] = n.og;
                    d.t[3] = w.g_pa;
                    d.i[0] = t;
                    d.i[1] = c.hidden;
                    d.f[0] = c.eps;
                    d.f[1] = 1.0;
                })
            } else {
                b.emit(DevOp::Residual, elem(t * c.hidden), &[c_o], |d| {
                    d.t[0] = n.x;
                    d.t[1] = n.x;
                    d.t[2] = n.og;
                    d.i[0] = t * c.hidden;
                    d.f[0] = 1.0;
                })
            };
            let pre_mlp_norm = if gemma { w.g_pf } else { w.g_pa };
            b.emit(DevOp::RmsNorm, rows.clone(), &[c_r1], |d| {
                d.t[0] = n.hn;
                d.t[1] = n.x;
                d.t[2] = pre_mlp_norm;
                d.i[0] = t;
                d.i[1] = c.hidden;
                d.f[0] = c.eps;
            })
        };
        let (mlp_src, mlp_g) = (n.hn, TENSOR_NONE);
        // GATE|UP AS ONE GEMV WITH A FUSED GLU EPILOGUE -- the fusion every BLAS ships.
        //
        // gate and up read the same x and have the same shape, so one GEMV can compute BOTH
        // halves of its own output columns and apply act(gate)*up as it writes them. The GLU is
        // then output-stationary: the workgroup owning column n is the only one that touches it,
        // so the GLU runs exactly once per element and NOTHING is replicated. Three packets
        // (gemv, gemv, glu) collapse to one, and the GLU's global gate -- which 250 of 256 CUs
        // stalled behind, it ran on 6 workgroups -- disappears with it.
        //
        // The DIRECTION is the whole thing. Folding the GLU into the *down* GEMV's LDS staging
        // instead (the consumer's PROLOGUE) was measured at a 39x LOSS: `fu` is down's K
        // dimension, so all 256 of its workgroups stage the whole of it and each recomputes the
        // entire GLU. Fuse into the producer's EPILOGUE, never into the consumer's PROLOGUE.
        //
        // Needs x staged on-chip (its A-operand is read once per output column), so it is a
        // decode-path op; prefill keeps the tiled GEMM triple, where the GLU amortises anyway.
        // Prefill fuses too, via the GEMM epilogue (DevOp::GemmGlu) -- same fusion, same law.
        // The GEMV form needs x staged on-chip; the GEMM form has no such constraint, it just
        // stages a different B tile. Requires the 256x256 tile (its SN axis is what carries
        // gate-vs-up), so only when pick_tile would have chosen Gemm anyway.
        // gate/up are COLUMN-parallel (inter_l lanes on this rank); the GLU is elementwise on the
        // rank's own lanes, so no communication. `c_gl` is the dependency feeding down_proj.
        let glu_fused = gemv_family && (t as u64 * c.hidden as u64) <= GM_LDS_HALVES;
        let gemm_glu = !gemv_family && pick_tile(t, inter_l, c.hidden, n_cu) == DevOp::Gemm;
        // w8a8: quant the (hidden-width) pre-FF norm output feeding gate/up. Reuses xqh/ash (q/k/v
        // already consumed them; the c_pf→o_proj→flash→qkv chain serializes the reuse). Inert
        // (returns c_pf) off the w8a8 path, so glu_fused/bf16 arms below keep their c_pf dep.
        // P9 hoist: emit the MoE router (score + topk) BEFORE the dense-MLP packets. Streams
        // execute in emission order per block, so with the router emitted after the MLP the
        // score/topk blocks only reach it once their dense slices retire — serializing
        // dense + router + expert-GLU. Hoisting turns that into max(dense, router) + GLU.
        // The DAG (deps: c_pf only) is unchanged; only stream position moves.
        let c_rt_hoist = if c.moe && decode {
            let root = (c.hidden as f32).powf(-0.5);
            Some(emit_gemma_moe_router(
                b,
                c_pf,
                n.moe_tab,
                n.x,
                w.rproj,
                w.rscale,
                w.rpes,
                n.moe_rscore,
                c.hidden,
                c.n_exp,
                c.top_k,
                root,
                c.eps,
                gemma_moe_router_split_plan(n_cu, c.n_exp, t),
                t,
            ))
        } else {
            None
        };
        let dmlp = quant(b, n.xqh, n.ash, mlp_src, c.hidden, c_pf);
        let c_gl = if glu_fused {
            // FP8 decode: gate|up fused GEMV+GLU on fp8 weights, each with its own dequant scale.
            if fp8 {
                b.emit(DevOp::GemvGluFp8, all.clone(), &[c_pf], |d| {
                    d.t[0] = n.fu;
                    d.t[1] = mlp_src;
                    d.t[2] = w.wg8;
                    d.t[5] = w.wu8;
                    d.t[3] = w.sg;
                    d.t[4] = w.su;
                    d.i[0] = t;
                    d.i[1] = inter_l;
                    d.i[2] = c.hidden;
                    d.i[5] = c.mlp_act; // 0 GeGLU (Gemma), 1 SwiGLU (Llama/Qwen)
                })
            } else {
                b.emit(DevOp::GemvGlu, all.clone(), &[c_pf], |d| {
                    d.t[0] = n.fu;
                    d.t[1] = mlp_src;
                    d.t[2] = w.wg;
                    d.t[5] = w.wu;
                    d.i[0] = t;
                    d.i[1] = inter_l;
                    d.i[2] = c.hidden;
                    d.i[5] = c.mlp_act; // 0 GeGLU (Gemma), 1 SwiGLU (Llama/Qwen)
                })
            }
        } else if gemm_glu && fp8 {
            // PREFILL fp8 GLU: w8a16 (default cubin) OR w8a8 (PLOW_NV_W8A8 cubin), same GEMM_GLU_FP8
            // opcode. w8a16: A bf16 (t1=mlp_src), Wg/Wu e4m3 (t2/t5), per-channel g/u scales (t4/t6).
            // w8a8: A e4m3 (t1=xqh) + per-row a_scale (t3=ash); Wg/Wu e4m3 + g/u scales — the
            // epilogue folds a_scale*sg (and a_scale*su) into both streams. Same fusion law.
            b.emit(DevOp::GemmGluFp8, all.clone(), &[dmlp], |d| {
                d.t[0] = n.fu;
                d.t[2] = w.wg8;
                d.t[5] = w.wu8;
                d.t[4] = w.sg;
                d.t[6] = w.su;
                if w8a8 {
                    d.t[1] = n.xqh;
                    d.t[3] = n.ash;
                } else {
                    d.t[1] = mlp_src;
                }
                d.i[0] = t;
                d.i[1] = inter_l;
                d.i[2] = c.hidden;
                d.i[5] = c.mlp_act; // 0 GeGLU (Gemma), 1 SwiGLU (Llama/Qwen)
            })
        } else if gemm_glu {
            b.emit(DevOp::GemmGlu, all.clone(), &[c_pf], |d| {
                d.t[0] = n.fu;
                d.t[1] = mlp_src;
                d.t[2] = w.wg;
                d.t[5] = w.wu;
                d.i[0] = t;
                d.i[1] = inter_l;
                d.i[2] = c.hidden;
                d.i[5] = c.mlp_act; // 0 GeGLU (Gemma), 1 SwiGLU (Llama/Qwen)
            })
        } else {
            // gate and up: same argument as q/k/v -- independent, so disjoint CU sets.
            let (cg, cu) = if gemv_family {
                split2(n_cu, 1, 1)
            } else {
                split2(n_cu, tiles(t, inter_l), tiles(t, inter_l))
            };
            let c_g = proj(
                b,
                n.gt,
                mlp_src,
                w.wg,
                w.wg8,
                w.sg,
                n.xqh,
                n.ash,
                t,
                inter_l,
                c.hidden,
                mlp_g,
                cg,
                &[dmlp],
            );
            let c_u = proj(
                b,
                n.ut,
                mlp_src,
                w.wu,
                w.wu8,
                w.su,
                n.xqh,
                n.ash,
                t,
                inter_l,
                c.hidden,
                mlp_g,
                cu,
                &[dmlp],
            );
            b.emit(DevOp::Glu, elem(t * inter_l), &[c_g, c_u], |d| {
                d.t[0] = n.fu;
                d.t[1] = n.gt;
                d.t[2] = n.ut;
                d.i[0] = t * inter_l;
                d.i[1] = c.mlp_act; // 0 GeGLU (Gemma), 1 SwiGLU
            })
        };
        // down_proj is ROW-parallel (input = inter_l lanes) → a PARTIAL H-vector. Under TP it
        // writes dg_tp and an XReduce sums the N peers into `dg` — all-reduce #2 of the layer,
        // at the second NormResidual boundary (plans/tp-design.md §3a, §8a). proj() picks the fp8
        // (GemvFp8) arm on the decode fp8 path via the wd8/sd operands.
        // w8a8: quant the (inter-width) GLU output feeding down_proj.
        let dfu = quant(b, n.xqi, n.asi, n.fu, inter_l, c_gl);
        let c_d = if tp > 1 {
            let c_dp = proj(
                b,
                n.dg_tp,
                n.fu,
                w.wd,
                w.wd8,
                w.sd,
                n.xqi,
                n.asi,
                t,
                c.hidden,
                inter_l,
                TENSOR_NONE,
                all.clone(),
                &[dfu],
            );
            emit_xreduce(
                b, &mut xgate, decode, &xr_cus, c_dp, n.dg, xr_elems, tp, slot_b,
            )
        } else {
            proj(
                b,
                n.dg,
                n.fu,
                w.wd,
                w.wd8,
                w.sd,
                n.xqi,
                n.asi,
                t,
                c.hidden,
                inter_l,
                TENSOR_NONE,
                all.clone(),
                &[dfu],
            )
        };
        // ===== Gemma-4 26B-A4B MoE branch (decode, B=1; plans/rtx-08-gemma4-moe-26b.md) =====
        // The dense MLP above produced `n.dg`. The MoE block adds a routed-expert branch and sums
        // the two through the sandwich: combined = post_ffn_norm(post_ffn_1(dense) + post_ffn_2(moe)).
        // Router & experts both read the RESIDUAL (n.x, set by c_pf), NOT the pre-MLP norm. The
        // second residual below then consumes `ffn_out` (= moe_comb) instead of n.dg.
        // P9 op72: when the MoE tail is fused (combine+resid+next-norm in one packet), this
        // carries its counter and the SECOND RESIDUAL block below is skipped entirely.
        let mut moe_fused_tail: Option<u32> = None;
        let (ffn_out, c_d) = if c.moe {
            let root = (c.hidden as f32).powf(-0.5);
            // BATCH>1 DECODE: `t` IS the batch B here. Every decode MoE op carries B in a spare
            // immediate, emitted only when B>1 so the B=1 packet stays byte-identical. The routed
            // work space becomes B*k slots; the kernels sweep it CHANNEL-MAJOR so slots that share
            // an expert read that expert's weight rows once from HBM (op_moe.cuh ordering note).
            let nb = if decode && t > 1 { t } else { 0 };
            assert!(
                !decode || t <= 32,
                "MoE decode batch is capped at 32 (per-CTA inv[] scratch, PLOW_MOE_MAXB)"
            );
            if decode {
            // h1 = post_feedforward_layernorm_1(dense MLP output)
            let c_h1 = b.emit(DevOp::RmsNorm, rows.clone(), &[c_d], |d| {
                d.t[0] = n.moe_h1;
                d.t[1] = n.dg;
                d.t[2] = w.g_pf1;
                d.i[0] = t;
                d.i[1] = c.hidden;
                d.f[0] = c.eps;
            });
            // router(residual): weightless-rms -> ·scale·root -> softmax -> top-k (lowest-id tie)
            // -> norm_topk -> ·per_expert_scale -> routing_table[k]. The default remains the
            // historical one-block opcode. The opt-in split scores eight experts per CTA, then a
            // one-CTA tail performs the exact serial softmax/top-k/gate ordering.
            let c_rt = c_rt_hoist.expect("router hoisted before dense MLP for moe");
            let _ = root;
            // Expert gate/up (fused) -> mfu[k,I]; expert down + gate scale -> part[k,H].
            // GluNorm fusion (op 71): fuse the pre-feedforward-norm-2 INTO the expert GLU,
            // eliminating a separate RmsNorm op + counter gate. Each CTA redundantly computes
            // the RMS of the residual (5.6 KB @ H=2816, hot in L2 from the router read).
            // Falls back to separate norm + GLU when fp8 (no fused fp8 variant yet).
            let glu_cus: Vec<u32> = (0..n_cu).collect();
            let down_cus: Vec<u32> = (0..n_cu).collect();
            let glu_op = if fp8 {
                DevOp::MoeExpertGluGemmaFp8
            } else {
                DevOp::MoeExpertGluNormGemma
            };
            let c_glu = if fp8 {
                // fp8 path: separate norm + expert GLU (no fused fp8 norm variant)
                let c_xn2_local = b.emit(DevOp::RmsNorm, rows.clone(), &[c_pf], |d| {
                    d.t[0] = n.moe_xn2;
                    d.t[1] = n.x;
                    d.t[2] = w.g_pre2;
                    d.i[0] = t;
                    d.i[1] = c.hidden;
                    d.f[0] = c.eps;
                });
                b.emit(DevOp::MoeExpertGluGemmaFp8, glu_cus, &[c_rt, c_xn2_local], |d| {
                    d.t[0] = n.moe_mfu;
                    d.t[1] = n.moe_xn2;
                    d.t[2] = n.moe_tab;
                    d.t[3] = w.ewt;
                    d.t[4] = w.est;
                    d.i[0] = c.top_k;
                    d.i[1] = c.moe_inter;
                    d.i[2] = c.hidden;
                    d.i[3] = c.n_exp;
                    d.i[5] = nb; // BATCH B (0 at B=1: byte-identical)
                })
            } else {
                // bf16 path: fused norm + expert GLU (one fewer gate)
                b.emit(DevOp::MoeExpertGluNormGemma, glu_cus, &[c_rt, c_pf], |d| {
                    d.t[0] = n.moe_mfu;
                    d.t[1] = n.x;
                    d.t[2] = n.moe_tab;
                    d.t[3] = w.ewt;
                    d.t[4] = w.g_pre2;
                    d.i[0] = c.top_k;
                    d.i[1] = c.moe_inter;
                    d.i[2] = c.hidden;
                    d.i[3] = c.n_exp;
                    d.i[5] = nb; // BATCH B (0 at B=1: byte-identical)
                    d.f[0] = c.eps;
                })
            };
            let down_op = if fp8 {
                DevOp::MoeExpertDownGemmaFp8
            } else {
                DevOp::MoeExpertDownGemma
            };
            let c_dn = vec![b.emit(down_op, down_cus, &[c_glu], |d| {
                d.t[0] = n.moe_part;
                d.t[1] = n.moe_mfu;
                d.t[2] = n.moe_tab;
                d.t[3] = w.ewt;
                d.t[4] = w.est;
                d.i[0] = c.top_k;
                d.i[1] = c.hidden;
                d.i[2] = c.moe_inter;
                d.i[3] = c.n_exp;
                d.i[5] = nb; // BATCH B (0 at B=1: byte-identical)
            })];
            // fused combine + rmsnorm + residual: saves 2 counter gates per layer.
            let mut comb_deps: Vec<u32> = c_dn;
            comb_deps.push(c_h1);
            // op72 MEASURED NEGATIVE in its scalar form (P9, 2026-07-20): +0.18 ms/token on
            // BOTH bf16 (8.04→8.22) and fp8 (6.03→6.21) @40ctx — the 1-block 4-pass scalar
            // body costs more than the packet boundary it removes, and its reduction order
            // differs from the vectorized NormResidualNorm (last-ulp bf16 flips → token
            // drift vs the pair). Oracle is bit-exact vs its own golden. Default OFF; only
            // worth revisiting as a register-cached vectorized body that replicates NRN's
            // summation order. Opt in: PLOW_GEMMA_MOE_TAIL_FUSE=1.
            let tail_fuse = std::env::var("PLOW_GEMMA_MOE_TAIL_FUSE").ok().as_deref()
                == Some("1");
            // op72 is a single-row 1-CTA body and is default-OFF (measured negative); it was not
            // batched. Refuse the combination loudly rather than emit wrong rows 1..B.
            assert!(
                !(tail_fuse && t > 1),
                "PLOW_GEMMA_MOE_TAIL_FUSE is B=1 only (op72 MoeCombineResidNormGemma is not batched)"
            );
            let c_comb = if gfuse && tail_fuse {
                // op72: fused combine + post_ffn norm + sandwich residual + NEXT input norm.
                // One 1-block packet replaces the (op70, NormResidualNorm) pair on the layer
                // tail — the chain next-QKV gates on loses a packet boundary. Bit-exact.
                let next_gin = if l + 1 < block.end {
                    n.lw[l + 1].g_in
                } else {
                    n.fin
                };
                let ct = b.emit(DevOp::MoeCombineResidNormGemma, vec![0], &comb_deps, |d| {
                    d.t[0] = n.hn;
                    d.t[1] = n.x;
                    d.t[2] = n.moe_part;
                    d.t[3] = n.moe_h1;
                    d.t[4] = w.g_pf2;
                    d.t[5] = w.g_po;
                    d.t[6] = next_gin;
                    d.i[0] = c.hidden;
                    d.i[1] = c.top_k;
                    d.f[0] = c.eps;
                    d.f[1] = ls[l];
                });
                moe_fused_tail = Some(ct);
                ct
            } else {
                // BATCH B>1: one CTA per row (the body is a per-row block loop).
                let comb_cus: Vec<u32> = (0..t).collect();
                b.emit(DevOp::MoeCombineNormGemma, comb_cus, &comb_deps, |d| {
                    d.t[0] = n.moe_comb;
                    d.t[1] = n.moe_part;
                    d.t[2] = n.moe_h1;
                    d.t[3] = w.g_pf2;
                    d.i[0] = c.hidden;
                    d.i[1] = c.top_k;
                    d.i[2] = nb; // BATCH B (0 at B=1: byte-identical)
                    d.f[0] = c.eps;
                })
            };
            (n.moe_comb, c_comb)
            } else {
            // ===== GROUPED-MoE PREFILL (T rows; plans/p9-26b-prefill-moe.md) =====
            // h1 = post_ffn_norm_1(dense), T rows. xn2 = pre_ffn_norm_2(residual), T rows.
            let c_h1 = b.emit(DevOp::RmsNorm, rows.clone(), &[c_d], |d| {
                d.t[0] = n.moe_h1;
                d.t[1] = n.dg;
                d.t[2] = w.g_pf1;
                d.i[0] = t;
                d.i[1] = c.hidden;
                d.f[0] = c.eps;
            });
            let c_xn2 = b.emit(DevOp::RmsNorm, rows.clone(), &[c_pf], |d| {
                d.t[0] = n.moe_xn2;
                d.t[1] = n.x;
                d.t[2] = w.g_pre2;
                d.i[0] = t;
                d.i[1] = c.hidden;
                d.f[0] = c.eps;
            });
            // T-token router -> routing_table[T*k] (block-per-token, bit-identical to decode).
            let c_rt = b.emit(DevOp::MoeRouterGemmaPf, all.clone(), &[c_pf], |d| {
                d.t[0] = n.moe_tab;
                d.t[1] = n.x;
                d.t[2] = w.rproj;
                d.t[3] = w.rscale;
                d.t[4] = w.rpes;
                d.i[0] = c.hidden;
                d.i[1] = c.n_exp;
                d.i[2] = c.top_k;
                d.i[3] = t;
                d.f[0] = root;
                d.f[1] = c.eps;
            });
            // align/sort (SINGLE block): histogram -> padded prefix -> scatter gather maps.
            let c_align = b.emit(DevOp::MoeAlignGemmaPf, vec![0], &[c_rt], |d| {
                d.t[0] = n.moe_meta;
                d.t[1] = n.moe_tab;
                d.t[2] = n.moe_rowtok;
                d.t[3] = n.moe_rowpart;
                d.t[4] = n.moe_rowgate;
                d.i[0] = t;
                d.i[1] = c.n_exp;
                d.i[2] = c.top_k;
            });
            // grouped gate/up GEMM + GeGLU (gathered A, expert-selected B) -> fu_gathered.
            // beat26b: w8a8 arm = native fp8 tensor-core GEMM (both operands e4m3). xn2 is quantized
            // to e4m3 (xqh/ash, hidden width) once; the grouped GLU gathers e4m3 rows and dequants
            // with a_scale[token]*w_scale[chan] in the epilogue. bf16 arm unchanged.
            let c_dn = if w8a8 {
                // total_pad rows the align op touched for THIS bucket (matches align's write extent).
                let moe_total_pad = t * c.top_k + c.n_exp * 128;
                let c_xn2q = quant(b, n.xqh, n.ash, n.moe_xn2, c.hidden, c_xn2);
                let c_glu = b.emit(DevOp::MoeGroupGluGemmaPfW8a8, all.clone(), &[c_align, c_xn2q], |d| {
                    d.t[0] = n.moe_fug;
                    d.t[1] = n.xqh;      // xn2 e4m3
                    d.t[2] = w.ewt;      // fp8 expert weights
                    d.t[3] = n.moe_meta;
                    d.t[4] = n.moe_rowtok;
                    d.t[5] = n.ash;      // per-token a_scale
                    d.t[6] = w.est;      // per-channel weight scales
                    d.i[0] = c.moe_inter;
                    d.i[1] = c.hidden;
                    d.i[2] = c.n_exp;
                    d.i[5] = c.mlp_act;
                });
                // quant the gathered GLU output (total_pad rows, moe_inter width) for the down GEMM.
                let c_fuq = b.emit(DevOp::QuantFp8, all.clone(), &[c_glu], |d| {
                    d.t[0] = n.moe_fuq;
                    d.t[1] = n.moe_fug;
                    d.t[2] = n.moe_fus;
                    d.i[0] = moe_total_pad;
                    d.i[1] = c.moe_inter;
                });
                b.emit(DevOp::MoeGroupDownGemmaPfW8a8, all.clone(), &[c_fuq, c_align], |d| {
                    d.t[0] = n.moe_part;
                    d.t[1] = n.moe_fuq;  // fu e4m3
                    d.t[2] = w.ewt;
                    d.t[3] = n.moe_meta;
                    d.t[4] = n.moe_rowpart;
                    d.t[5] = n.moe_rowgate;
                    d.t[6] = w.est;
                    d.t[7] = n.moe_fus;  // per-row fu scale
                    d.i[0] = c.hidden;
                    d.i[1] = c.moe_inter;
                    d.i[2] = c.n_exp;
                })
            } else {
            let c_glu = b.emit(DevOp::MoeGroupGluGemmaPf, all.clone(), &[c_align, c_xn2], |d| {
                d.t[0] = n.moe_fug;
                d.t[1] = n.moe_xn2;
                d.t[2] = w.ewt;
                d.t[3] = n.moe_meta;
                d.t[4] = n.moe_rowtok;
                d.i[0] = c.moe_inter;
                d.i[1] = c.hidden;
                d.i[2] = c.n_exp;
                d.i[5] = c.mlp_act; // 0 GeGLU (Gemma)
            });
            // grouped down GEMM + gate-scale + scatter -> part[T,k,H].
            b.emit(DevOp::MoeGroupDownGemmaPf, all.clone(), &[c_glu, c_align], |d| {
                d.t[0] = n.moe_part;
                d.t[1] = n.moe_fug;
                d.t[2] = w.ewt;
                d.t[3] = n.moe_meta;
                d.t[4] = n.moe_rowpart;
                d.t[5] = n.moe_rowgate;
                d.i[0] = c.hidden;
                d.i[1] = c.moe_inter;
                d.i[2] = c.n_exp;
            })
            };
            // T-row combine + sandwich: out[t] = RMSNorm(Σ_slot part[t][slot], g_pf2) + h1[t].
            let c_comb = b.emit(DevOp::MoeCombineNormGemmaPf, all.clone(), &[c_dn, c_h1], |d| {
                d.t[0] = n.moe_comb;
                d.t[1] = n.moe_part;
                d.t[2] = n.moe_h1;
                d.t[3] = w.g_pf2;
                d.i[0] = c.hidden;
                d.i[1] = c.top_k;
                d.i[2] = t;
                d.f[0] = c.eps;
            });
            (n.moe_comb, c_comb)
            }
        } else {
            (n.dg, c_d)
        };
        // SECOND RESIDUAL.
        //   Gemma: x = (x + post_ffn_norm(d)) * layer_scalar — the learned scalar folds in.
        //   Llama/Qwen: x = x + d (plain).
        dep = if let Some(ct) = moe_fused_tail {
            // op72 already produced the new residual (n.x) AND the next input norm (n.hn).
            ct
        } else if fuse_norm {
            // x += down; then normalise for the NEXT sublayer's attention (the next layer's
            // input_layernorm, or the model's final norm after the last layer). One packet does
            // the end-of-layer residual AND the next input norm, so the loop top skips c_n.
            let next_gin = if l + 1 < block.end {
                n.lw[l + 1].g_in
            } else {
                n.fin
            };
            b.emit(DevOp::AddNorm, rows.clone(), &[c_d], |d| {
                d.t[0] = n.hn;
                d.t[1] = n.x;
                d.t[2] = n.x;
                d.t[3] = ffn_out;
                d.t[4] = next_gin;
                d.i[0] = t;
                d.i[1] = c.hidden;
                d.f[0] = c.eps;
            })
        } else if gfuse {
            // x = (x + post_ffn_norm(down)) * layer_scalar; hn = input_norm(x) for the NEXT layer
            // (or the final norm after the last layer). One packet does the end-of-layer sandwich
            // residual AND the next input norm, so the loop top skips c_n (same as fuse_norm).
            let next_gin = if l + 1 < block.end {
                n.lw[l + 1].g_in
            } else {
                n.fin
            };
            b.emit(DevOp::NormResidualNorm, rows.clone(), &[c_d], |d| {
                d.t[0] = n.hn;
                d.t[1] = n.x;
                d.t[2] = n.x;
                d.t[3] = ffn_out;
                d.t[4] = w.g_po;
                d.t[5] = next_gin;
                d.i[0] = t;
                d.i[1] = c.hidden;
                d.f[0] = c.eps;
                d.f[1] = ls[l];
            })
        } else if gemma {
            b.emit(DevOp::NormResidual, rows.clone(), &[c_d], |d| {
                d.t[0] = n.x;
                d.t[1] = n.x;
                d.t[2] = ffn_out;
                d.t[3] = w.g_po;
                d.i[0] = t;
                d.i[1] = c.hidden;
                d.f[0] = c.eps;
                d.f[1] = ls[l];
            })
        } else {
            b.emit(DevOp::Residual, elem(t * c.hidden), &[c_d], |d| {
                d.t[0] = n.x;
                d.t[1] = n.x;
                d.t[2] = ffn_out;
                d.i[0] = t * c.hidden;
                d.f[0] = 1.0;
            })
        };
    }

    // BLOCK MODE: stop here. `act.x` (n.x) — the post-FFN residual the loop's
    // last layer wrote — IS the block output the harness downloads. The final
    // norm, lm_head, softcap and argmax tail belongs to the whole-model run; a
    // single block emits neither logits nor a sampled token (act.logits stays
    // declared-but-unwritten, satisfying GpuEngine's mandatory-handle check).
    if block_mode {
        return;
    }

    // In the fused path the last layer's end-of-layer fused norm already applied the FINAL norm
    // (its next_gin was n.fin), so n.hn holds the final-normed row and c_f is just that dep.
    let c_f = if fuse_norm || gfuse {
        dep
    } else {
        b.emit(DevOp::RmsNorm, rows.clone(), &[dep], |d| {
            d.t[0] = n.hn;
            d.t[1] = n.x;
            d.t[2] = n.fin;
            d.i[0] = t;
            d.i[1] = c.hidden;
            d.f[0] = c.eps;
        })
    };
    let head_src = n.hn;
    // lm_head over the LAST row only (i4 = a_row0). Weight is the tied embedding table, or the
    // separate lm_head.weight when the checkpoint does not tie (Llama).
    let head_w = if c.tied { n.emb } else { n.head };
    // lm_head is COLUMN(vocab)-parallel (plans/tp-design.md §3a/§8d): each rank produces its
    // vocab_l logit lanes. tp-host binds the rank's vocab slice of the (replicated) weight.
    let lm_op = if gemv_family {
        DevOp::Gemv
    } else {
        pick_tile(1, vocab_l, c.hidden, n_cu)
    };
    // PREFILL takes only the LAST prompt row's logits (M=1, a_row0=t-1). DECODE takes ALL t rows,
    // one per sequence (M=t, a_row0=0) — batch>1 samples a token per sequence. Decode B=1 gives
    // (M=1, a_row0=0), identical to the old (1, t-1) since t==1 there.
    let (lm_m, lm_row0) = if decode { (t, 0) } else { (1, t - 1) };
    // PLOW_FP8_HEAD: weight-only fp8 lm_head (GemvFp8, dequant-on-load, per-row scale).
    // The tied embedding LOOKUP stays bf16 (reads the original table); only the head GEMV
    // reads the fp8 twin. Own reporting row — vLLM's fp8 recipe keeps lm_head bf16.
    let fp8_head = decode && n.head8 != TENSOR_NONE;
    // E5 (rtx-19) PLOW_FUSE_ARGMAX: fuse the greedy-argmax epilogue (+ softcap) into the lm_head
    // GEMV, folding each block's owned vocab slice into an amax partial and dropping the SoftCap +
    // Argmax packets. Greedy B=1 decode on the bf16 head only (fp8 head keeps the classic path).
    let fuse_am = fuse_argmax_on() && decode && gemv_family && !fp8_head && t == 1;
    let lm_op = if fuse_am {
        DevOp::GemvArgmax
    } else if fp8_head {
        DevOp::GemvFp8
    } else {
        lm_op
    };
    let c_lm = b.emit(lm_op, all.clone(), &[c_f], |d| {
        d.t[0] = n.logits;
        d.t[1] = head_src;
        d.t[2] = if fp8_head { n.head8 } else { head_w };
        if fp8_head {
            d.t[5] = n.head8s;
        }
        if fuse_am {
            d.t[3] = n.amax; // packed-u64 partials, one per block
            d.f[0] = c.softcap; // reproduced in the epilogue; 0 = none
        }
        d.i[0] = lm_m;
        d.i[1] = vocab_l;
        d.i[2] = c.hidden;
        d.i[3] = 0;
        d.i[4] = lm_row0;
    });
    // Final-logit softcap: Gemma only (cap 30). Llama/Qwen have none, and d_softcap divides by
    // cap, so it must be SKIPPED (not emitted with cap 0) for them. Fused into GemvArgmax above.
    let c_logits = if fuse_am {
        c_lm
    } else if c.softcap > 0.0 {
        // BATCH>1: softcap the [t][vocab] logit tile (flat t*vocab). B=1 => vocab_l, identical.
        b.emit(DevOp::SoftCap, elem(lm_m * vocab_l), &[c_lm], |d| {
            d.t[0] = n.logits;
            d.t[1] = n.logits;
            d.i[0] = lm_m * vocab_l;
            d.f[0] = c.softcap;
        })
    } else {
        c_lm
    };

    // Greedy sample on device, and write the id into `in.ids` -- the very tensor the NEXT
    // step's EMBED reads. The host never sees the 512 KB logit row: it reads 4 bytes to print
    // the token and to check for EOS, and writes nothing back.
    // BATCH>1: i1 = n_batch. Each sequence argmaxes its OWN [vocab] row into amax[b][*] and
    // ArgmaxFin folds it into ids[b] — one token per sequence, no cross-sequence bleed. i1==0
    // (B=1/prefill) is the single-sequence path, byte-identical.
    // `decode` guard: in a PREFILL program t is the BUCKET SIZE (128..8192), not a batch —
    // without the guard every prefill bucket emitted argmax over t "sequences", reading
    // t*vocab logits from the [dbatch][vocab] tensor (a 64 MiB OOB read at t=8192) and
    // clobbering ids[0..t]. Prefill is always single-sequence (lm_head M=1 → logits row 0).
    let nb_argmax = if decode && t > 1 { t } else { 0 };
    // FUSED (fuse_am): GemvArgmax already wrote the `all.len()` partials — skip the Argmax packet
    // and fold that many. CLASSIC: the 64-block Argmax strides the full vocab, folding AMAX_BLOCKS.
    let (c_am, nparts) = if fuse_am {
        (c_lm, all.len() as u32)
    } else {
        let amax_cus: Vec<u32> = (0..AMAX_BLOCKS).collect();
        let c_am = b.emit(DevOp::Argmax, amax_cus, &[c_logits], |d| {
            d.t[0] = n.amax;
            d.t[1] = n.logits;
            d.i[0] = vocab_l;
            d.i[1] = nb_argmax;
        });
        (c_am, AMAX_BLOCKS)
    };
    let c_fin = b.emit(DevOp::ArgmaxFin, vec![0], &[c_am], |d| {
        d.t[0] = n.ids;
        d.t[1] = n.amax;
        d.i[0] = nparts;
        d.i[1] = nb_argmax;
    });
    // lm_head is REPLICATED under TP (see declare() note): every rank computes the full-vocab
    // argmax and thus the SAME global token id, so no cross-rank XArgmaxFin fold is needed. The
    // sharded lm_head + XArgmaxFin id-fold is a Phase-3 item (§8d, §13); c_fin already wrote the
    // correct global id into in.ids on every rank.
    let _ = c_fin;
}

/// Blocks the argmax partial reduction is spread over. 64 x 512 threads covers a 262144-entry
/// vocab in one strided pass per thread.
const AMAX_BLOCKS: u32 = 64;

/// E5 (rtx-19): PLOW_FUSE_ARGMAX fuses the greedy-argmax epilogue into the lm_head GEMV
/// (`DevOp::GemvArgmax`), replacing the `SoftCap` + `Argmax` packets. Default off → byte-identical.
fn fuse_argmax_on() -> bool {
    std::env::var("PLOW_FUSE_ARGMAX").ok().as_deref() == Some("1")
}

/// Argmax-partial slot count: when fused the lm_head runs on all `n_cu` blocks (one partial each),
/// so the buffer and `ArgmaxFin` fold must cover `max(AMAX_BLOCKS, n_cu)`; classic keeps AMAX_BLOCKS.
fn fuse_argmax_parts(n_cu: u32) -> u32 {
    if fuse_argmax_on() {
        n_cu.max(AMAX_BLOCKS)
    } else {
        AMAX_BLOCKS
    }
}

// ============================================================================
// GLM-5.2-FP8 (GlmMoeDsa) — MLA + DSA + block-fp8 MoE serving path.
//
// A DeepSeek-V3.2-class model: Multi-head Latent Attention (absorbed q_nope/value
// folds + partial INTERLEAVED RoPE on the 64 rope dims), a DSA "lightning indexer"
// (ctx>2048; a no-op below), and a fine-grained sigmoid-router block-fp8 MoE
// (256 routed experts, top-8, +e_score_correction_bias, norm_topk, route_scale 2.5,
// 1 shared expert). This is a WHOLLY SEPARATE emit path from the dense-GQA
// emit_phase above — the op set (FLASH_MLA_DECODE/O_UV_FOLD/MOE_*) and the derived
// weights (absorbed Wqa/Wuv) share nothing with Gemma/Llama/Qwen.
//
// The op sequence emit_glm_block produces is the EXACT 34-op block validated on
// gfx950 by runtime/tests/glm52_real_block_gfx950_test.c against the HF oracle
// (real 256 experts, real [128,128] block-fp8 scales) — see plans/glm52-campaign.md
// "B4-CORE DONE". The offline glm_tests below assert byte-for-op equality with that
// reference, so the emitted layer inherits the harness's passing GPU result.
//
// MILESTONE-1 STAGING (plans/glm52-campaign.md): the query/key RoPE is folded into
// the derived weights at a FIXED position by the host weight-prep (as the B4 harness
// did) — valid for single-token validation. The dynamic INTERLEAVED-RoPE op (coming
// from the kernels branch) replaces the fold for milestone-3 multi-token decode.
// ============================================================================

/// GLM-5.2 (GlmMoeDsa) config — parsed from the real `config.json`. Dims verified in
/// `plans/glm52-arch.md`. `H`/`NH`/`DK`(kv_lora)/`QL`(q_lora)/`QN`(qk_nope)/`DR`(qk_rope)/
/// `VD`(v_head) name the MLA geometry the kernels carry as compile-time operands.
#[derive(Clone)]
struct GlmCfg {
    layers: u32,        // 78 (layer 78 = MTP head, skipped)
    hidden: u32,        // H 6144
    heads: u32,         // NH 64
    kv_lora: u32,       // DK 512 (latent cache width)
    q_lora: u32,        // QL 2048
    qk_nope: u32,       // QN 192 (absorbed into the latent)
    qk_rope: u32,       // DR 64  (partial rope, interleaved)
    v_head: u32,        // VD 256
    vocab: u32,         // 154880
    eps: f32,           // 1e-5
    n_exp: u32,         // E 256 routed experts
    top_k: u32,         // 8
    moe_inter: u32,     // IMOE 2048 (per-expert intermediate)
    dense_inter: u32,   // 12288 (layers < first_k_dense)
    first_k_dense: u32, // 3 (layers 0,1,2 dense FFN; 3-77 MoE)
    route_scale: f32,   // 2.5 (routed_scaling_factor)
    attn_scale: f32,    // 1/sqrt(qk_head_dim = qk_nope+qk_rope = 256) = 0.0625
    rope_theta: f64,    // 8e6 (interleaved partial RoPE on the 64 rope dims)
    tp: u32,
    // EP (expert-parallel) over the same `tp` world: attention/shared/dense stay TP-sharded (the
    // "floor" is parallelized), but the ROUTED experts are distributed WHOLE across ranks (256/tp
    // per rank, full moe_inter width — no CU-starve) instead of TP-sliced. Each rank fires only its
    // LOCAL chosen experts (host binds local expert bases, NULL for remote; the kernel skips a null
    // base). The combine XReduce (already summing shared partials over tp) folds the per-rank whole-
    // expert partials in the SAME collective — no new op. See plans/moe-ep-kernels.md §5a.
    ep: bool,
    // Collapse the per-slot expert packets (2*top_k) into 2 grouped packets (ops 48/49) — the op-count
    // lever for M=1 decode. Bit-identical output; block-fp8 only.
    group: bool,
    // DSA lightning indexer (GlmMoeDsa). ctx>2048 => indexer->select->gather; ctx<=2048 => dense.
    index_heads: u32,        // index_n_heads = 32
    index_dim: u32,          // index_head_dim = 128 (rope on the first qk_rope=64; pass the rest)
    index_topk: u32,         // index_topk = 2048
    indexer_full: Vec<bool>, // per-layer: true='full' (owns an indexer), false='shared' (reuse last full)
    // Whether this arch HAS the DSA lightning indexer at all. GLM-5.2 (glm_moe_dsa) => true.
    // Kimi K2.7 / DeepSeek-V3 are plain MLA (NO indexer), so `has_dsa=false` holds the DSA gate off
    // at EVERY ctx — declare_glm allocates no indexer scratch and emit_glm_mla stays on FlashMlaDecode
    // (the dense MLA path), reusing the same emit as GLM below the crossover.
    has_dsa: bool,
}
impl GlmCfg {
    /// Full per-head qk width = nope + rope = 256. The attention softmax scale is
    /// 1/sqrt of THIS (0.0625) — NOT 1/sqrt(128); the absorbed MLA keeps the full-width scale.
    fn qk_head(&self) -> u32 {
        self.qk_nope + self.qk_rope
    }
    /// Layers `[0, first_k_dense)` are dense-FFN (intermediate 12288); the rest are MoE.
    fn is_dense(&self, layer: u32) -> bool {
        layer < self.first_k_dense
    }
    /// DSA gate: sparse (indexer->select->gather) only above the dense-attention CROSSOVER — the ctx
    /// where the gather's FIXED per-full-layer overhead (indexer score + top-k select on 21 layers)
    /// plus the constant top_k=2048 gather flash first UNDERCUTS the ctx-linear dense flash. MEASURED
    /// on the real full 78-layer model (TP4, MI350X 4-7, median-11) AFTER the MFMA-indexer + 32-WG-select
    /// interp wiring: gather tpot is flat ~48.6ms; dense grows ~0.136ms/1k-ctx from 41.4ms@16k, so the
    /// two cross at ~69k (BEFORE the wiring: ~91k). Below the crossover the whole-model tpot is
    /// MoE/projection-floor-dominated (~40ms) and dense-flash is cheap, so gather LOSES (0.85-0.90x
    /// across 16-32k) — those ctx are gated to dense, the measured winner. `CROSSOVER=65536` keeps the
    /// 16k-32k band (and up to 64k) on dense and arms gather only where it wins. NOTE: this is the TP4
    /// crossover (the session's GPU budget is 4 cards); a TP8 deployment halves the parallel floor and
    /// per-rank attention shrinks, lowering the crossover — recalibrate with an 8-GPU sweep before
    /// serving TP8 (design-doc projection puts the TP8 band nearer the crossover).
    /// PLOW_GLM_DSA=0 forces the dense path even at long ctx (the apples-to-apples decode baseline).
    fn dsa(&self, ctx: u32) -> bool {
        const CROSSOVER: u32 = 65536; // measured full-model TP4 dense/gather crossover (~69k, rounded)
        self.has_dsa
            && ctx > CROSSOVER
            && std::env::var("PLOW_GLM_DSA").ok().as_deref() != Some("0")
    }
    /// A 'full' indexer layer owns its own indexer; 'shared' layers reuse the last full layer's idx.
    fn indexer_is_full(&self, layer: u32) -> bool {
        self.indexer_full
            .get(layer as usize)
            .copied()
            .unwrap_or(false)
    }
}

fn cfg_glm(dir: &Path) -> GlmCfg {
    let v: Value =
        serde_json::from_slice(&std::fs::read(dir.join("config.json")).expect("config.json"))
            .unwrap();
    let g = |k: &str| {
        v[k].as_u64()
            .unwrap_or_else(|| panic!("config.json missing {k}")) as u32
    };
    let qk_head = g("qk_nope_head_dim") + g("qk_rope_head_dim");
    GlmCfg {
        layers: g("num_hidden_layers"),
        hidden: g("hidden_size"),
        heads: g("num_attention_heads"),
        kv_lora: g("kv_lora_rank"),
        q_lora: g("q_lora_rank"),
        qk_nope: g("qk_nope_head_dim"),
        qk_rope: g("qk_rope_head_dim"),
        v_head: g("v_head_dim"),
        vocab: g("vocab_size"),
        eps: v["rms_norm_eps"].as_f64().unwrap() as f32,
        n_exp: g("n_routed_experts"),
        top_k: g("num_experts_per_tok"),
        moe_inter: g("moe_intermediate_size"),
        dense_inter: g("intermediate_size"),
        first_k_dense: g("first_k_dense_replace"),
        route_scale: v["routed_scaling_factor"].as_f64().unwrap() as f32,
        attn_scale: (qk_head as f32).powf(-0.5),
        rope_theta: v["rope_theta"].as_f64().unwrap_or(8_000_000.0),
        tp: 1,
        ep: std::env::var("GLM_EP").ok().as_deref() == Some("1"),
        group: std::env::var("GLM_GROUP").ok().as_deref() == Some("1"),
        index_heads: v["index_n_heads"].as_u64().unwrap_or(32) as u32,
        index_dim: v["index_head_dim"].as_u64().unwrap_or(128) as u32,
        index_topk: v["index_topk"].as_u64().unwrap_or(2048) as u32,
        indexer_full: v["indexer_types"]
            .as_array()
            .map(|a| a.iter().map(|t| t.as_str() == Some("full")).collect())
            .unwrap_or_default(),
        has_dsa: true, // GLM-5.2 (glm_moe_dsa) has the DSA lightning indexer.
    }
}

/// Kimi K2.7 / DeepSeek-V2/V3 cfg (plans/block-asset-harness.md §5.0/§5.3, M3). These are plain
/// MLA + MoE — the SAME DeepSeek-derived config schema GLM uses (q/kv_lora, qk_nope/rope, v_head,
/// n_routed_experts, moe_intermediate_size, first_k_dense_replace, routed_scaling_factor) but with
/// NO DSA lightning indexer. So the cfg reuses `cfg_glm`'s parse verbatim and only forces the DSA
/// gate off (`has_dsa=false`): the indexer fields default (indexer_types absent => empty) and never
/// fire, so declare_glm / emit_glm_mla take the dense-MLA path at every ctx. This is the reuse
/// seam — Kimi is GLM-below-the-crossover with different dims. NOT `rewrite/kimi.rs` (that lowers to
/// the wire-packet backend `GpuEngine` cannot load; see plan §5.0).
fn cfg_kimi(dir: &Path) -> GlmCfg {
    let mut c = cfg_glm(dir);
    c.has_dsa = false;
    c
}

/// MLA head-fusion factor `d_flash_mla_decode<512,64,GF>`, chosen PER PACKET from the pkt's fixed
/// max_ctx and baked into FlashMlaDecode i[7] (the interp instantiates GF∈{2,4} and dispatches on
/// i[7]; LDS/registers are sized for the GF=4 max, so occ is unchanged). GLM's MLA latent is
/// HEAD-SHARED, so GF query heads re-stream the compact latent once per head-group => latent HBM
/// traffic ~ n_head/GF. TRADEOFF (measured, MI350X full-model TP4 decode): GF=4 CUTS long ctx
/// (128k 125 vs 140 ms/tok; 8k-32k 1.3-1.6x on the MLA chain) but ADDS split/merge overhead that
/// HURTS short ctx (1k 79 vs 58 ms/tok — the tiny 1k latent stream isn't worth the extra splits).
/// So: GF=2 for short-ctx pkts (preserve the router-split ~58ms@1k), GF=4 for long-ctx pkts.
/// PLOW_GLM_GF pins GF∈{2,4} (crossover sweeps). Crossover ~4k (see perf-data/glm52-plow-decode-tuned.json).
/// Long-ctx / MAX head-fusion factor. Matches the op_attention.h GLM_MLA_GF define (the interp
/// sizes the MLA-decode LDS + registers for this GF), and is the GF `glm_nsplit`'s chip-fill cap
/// assumes (the per-pkt glm_gf never exceeds it, so nsplit stays a safe over-estimate at GF=2).
const GLM_MLA_GF: u32 = 4;
const GLM_GF_CROSSOVER: u32 = 4096; // max_ctx <= this -> GF=2; else GF=8
fn glm_gf(ctx: u32) -> u32 {
    // GF=8 measured 1.5-1.9x faster than GF=4 at every ctx>=8192 (P2, plans/
    // mla-sm120-kernels.md §7): the NH/GF latent-reread cut dominates; merge is
    // GF-independent (nsplit unchanged) and 134 regs < the 225 megakernel cap so
    // occupancy is unaffected. nsplit still sized for GLM_MLA_GF=4 (a conservative
    // under-split at GF=8 -> slight chip under-fill only at batch=1; GF=8 wins there
    // anyway). PLOW_GLM_GF pins {2,4,8}.
    std::env::var("PLOW_GLM_GF")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .filter(|&v| v == 2 || v == 4 || v == 8)
        .unwrap_or(if ctx <= GLM_GF_CROSSOVER { 2 } else { 8 })
}
/// MLA flash-decode KV-split count, CTX-ADAPTIVE (mirrors Gemma's PLOW_NS_MUL/ABS). The flash
/// splits its work into `n_grp*nsplit = (heads/GF)*nsplit` items over 256 CUs; nsplit must fill
/// the machine (n_grp*nsplit >= n_cu) without over-splitting short contexts (FlashMerge crit-path
/// busy scales with nsplit). fill_base = ceil(n_cu/n_grp); halve it below 2k ctx. Env PLOW_GLM_NS
/// pins nsplit directly (occupancy sweeps).
///
/// `heads` MUST be the PER-RANK head count (nh_l = n_head/tp), NOT the global n_head. The kernel
/// runs this rank's nh_l head-shard, so its work-item count is (nh_l/GF)*nsplit, and the chip-fill
/// CAP is `fill = ceil(n_cu / (nh_l/GF))`. Sizing that cap from the global n_head (the pre-TP bug)
/// pinned it to tp=1's 16 head-groups => the FlashMlaDecode op ran on 32 of 256 CUs at tp=8.
///
/// The split count is NOT simply "fill the chip": MLA decode is latent-HBM-reread-bound, so more
/// splits => more CUs streaming the latent in parallel => the decode OP drops ~1/nsplit (measured:
/// tp=8 ctx-32k decode 154->56us, 2.77x, decode_eff 244->676 GB/s at full fill). BUT plow's
/// FlashMerge is a SEPARATE O(nsplit) pass, so past a point its growth (tp=8 merge 28->54us at
/// ns 16->128) eats the decode saving — full fill REGRESSES the decode CHAIN at mid ctx (tp=8 8k
/// chain 123->155us at ns 128). The cost optimum balances the two: d/dns[ latent/nsplit + k*nsplit ]
/// = 0 => nsplit grows with the latent stream (~ctx), capped at `fill` (chip) and `kv_tiles` (no
/// empty splits). Measured MI350X chain optima (mla_perf, tp4 & tp8): ns~16 up to 8k, ns~64 at 32k;
/// `ctx/512` floored at 16 reproduces them and yields the 32k win (tp8 242->165us 1.47x, tp4
/// 240->176us 1.36x) with no mid-ctx regression. tp=1 is fill-capped to 16 (already chip-full), so
/// byte-identical. See plans/glm-mla-flash-tuning.md and Plow.SplitK (the split reduction equals
/// the sequential sum for ANY nsplit; occupancy is monotone in the split count up to n_cu).
fn glm_nsplit(ctx: u32, heads: u32) -> u32 {
    /// KV rows staged per flash step (op_attention.h FA_BKV) — the KV-tile granularity a split
    /// divides. A split covering zero whole tiles writes -inf and is pure overhead (a launched
    /// workgroup + an extra O(nsplit) merge input), so nsplit is capped at the tile count.
    const FA_BKV: u32 = 32;
    /// Latent bytes per split at which the decode saving stops beating the O(nsplit) merge growth.
    /// ns scales as ctx/NS_PER (measured knee) below the fill cap.
    const NS_PER: u32 = 512;
    /// Split floor: below this the fixed decode overhead already dominates, so extra splits only
    /// add merge cost (measured: ns=16 is the chain optimum for ctx<=8k at every TP degree).
    const NS_FLOOR: u32 = 16;
    let n_grp = (heads / GLM_MLA_GF).max(1);
    let fill = ((256 + n_grp - 1) / n_grp).max(1); // chip-fill cap: splits to cover 256 CUs
    let kv_tiles = ctx.div_ceil(FA_BKV).max(1); // never split finer than there are KV tiles
                                                // ctx-scaled cost optimum, floored, then capped by the chip and the KV-tile count.
    let ns = (ctx / NS_PER).max(NS_FLOOR).min(fill).min(kv_tiles).max(1);
    std::env::var("PLOW_GLM_NS")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .filter(|&v| v >= 1)
        .unwrap_or(ns)
}
/// Router flags: bit0 sigmoid, bit1 norm_topk, bit2 apply e_score_correction_bias to SELECTION
/// only (DeepSeek/GLM noaux_tc). Mirrors FLAGS in the B4 harness.
const GLM_ROUTER_FLAGS: u32 = 1 | 2 | 4;
/// Expert/shared GLU activation = SiLU (SwiGLU). Mirrors ACT in the B4 harness.
const GLM_ACT_SILU: u32 = 1;

/// Per-layer GLM weights. Derived (absorbed / rope-folded) tensors are bf16 and named under a
/// `.derived.` segment the host weight-prep writes; the block-fp8 projections, router, and experts
/// keep their checkpoint names. `TENSOR_NONE` for the sub-block a layer does not have (dense vs MoE).
struct GlmLW {
    // MLA attention (bf16 norms + derived absorbed/rope-folded weights).
    gin: u32,   // input_layernorm
    qad: u32,   // q_a_proj (H->QL)
    gqa: u32,   // q_a_layernorm
    wqa: u32,   // DERIVED absorbed q_nope   [NH*DK, QL]
    wqr: u32, // DERIVED RAW q_rope down (q_b_rope, NOT folded) [NH*DR, QL]; RoPE applied dynamically
    ckvd: u32, // DERIVED kv_a latent down    [DK, H]
    gkva: u32, // kv_a_layernorm
    krotd: u32, // DERIVED RAW k_rope down (kv_a rope slice, NOT folded) [DR, H]; RoPE applied dynamically
    wuv: u32,   // DERIVED absorbed value       [NH*DK, VD]
    wo: u32,    // o_proj (NH*VD -> H)
    gpost: u32, // post_attention_layernorm
    // MoE (sparse layers): router + shared expert + the two loader-filled pointer tables.
    wr: u32,   // mlp.gate.weight [E,H] bf16
    bias: u32, // mlp.gate.e_score_correction_bias [E] f32
    shg: u32,  // shared_experts.gate_proj
    shu: u32,  // shared_experts.up_proj
    shd: u32,  // shared_experts.down_proj
    ewt: u32,  // expert_weight_table [E*3] u64 device ptrs (loader-filled from bound experts)
    est: u32,  // expert_scale_table  [E*3] u64 device ptrs (block-fp8 scale grids)
    // Dense FFN (layers < first_k_dense): block-fp8 gate/up/down + their weight_scale_inv grids.
    // TENSOR_NONE on MoE layers.
    dgate: u32,
    dgate_s: u32,
    dup: u32,
    dup_s: u32,
    ddown: u32,
    ddown_s: u32,
    // DSA lightning indexer (TENSOR_NONE except on 'full' layers with the DSA gate on).
    iwqb: u32, // indexer.wq_b.weight (fp8 [HI*DI, QL]) + iwqb_s scale grid
    iwqb_s: u32,
    iwk: u32, // indexer.wk.weight (fp8 [DI, H]) + iwk_s scale grid
    iwk_s: u32,
    iknw: u32, // indexer.k_norm.weight [DI] bf16
    iknb: u32, // indexer.k_norm.bias   [DI] bf16
    iwp: u32,  // indexer.weights_proj.weight [HI, H] bf16
}

/// The GLM tensor table. Decode-shaped activations (one row) + per-layer latent/rope caches +
/// per-layer weights. Prefill activations are a later (B-sweep) concern.
// ids/pos/emb/fin/head/logits/amax are the embed + lm_head scaffolding the SINGLE-layer gate
// declares but does not yet consume — the full 78-layer decode phase (next milestone) wires them.
#[allow(dead_code)]
struct GlmTn {
    ids: u32,
    pos: u32,
    kvlen: u32,
    cos: u32,
    sin: u32,
    emb: u32,
    fin: u32,
    head: u32,
    // MLA activations
    x: u32,
    xn: u32,
    qlr: u32,
    qlat: u32,
    ckvraw: u32,
    qa: u32,
    qrr: u32, // raw q_rope (pre-RoPE) [NH*DR]
    qr: u32,
    krr: u32, // raw k_rope (pre-RoPE) [DR]
    opart: u32,
    mlpart: u32,
    olat: u32,
    oat: u32,
    attn: u32,
    xmid: u32,
    xn2: u32,
    // MoE activations
    tab: u32,
    rlogit: u32, // router score-GEMV output [n_exp] bf16 (feeds MoeRouterTopk)
    shfu: u32,
    shared: u32,
    fu: u32,
    dfu: u32, // dense-FFN intermediate [dense_inter] (layers 0-2)
    part: u32,
    xnext: u32,
    logits: u32,
    amax: u32,
    // TP peer partials + zero residual (TENSOR_NONE at tp==1)
    og_tp: u32,
    dg_tp: u32,
    zero_h: u32,
    // DSA indexer (TENSOR_NONE when the DSA gate is off). qidx/kidx_raw/kidx_normed/widx are per-step
    // scratch; iscore/iidx/ighist/igctl are the score+select scratch (shared across layers, sequential);
    // icos/isin are the [ctx][DI/2] identity-tail interleaved-RoPE tables (first qk_rope/2 real, rest 1/0).
    qidx: u32,        // rope'd indexer query [HI*DI]
    kidx_raw: u32,    // wk @ xn [DI] (pre-norm)
    kidx_normed: u32, // k_norm(kidx_raw) [DI] (pre-rope)
    widx: u32,        // weights_proj @ xn [HI]
    iscore: u32,      // f32 [ctx] indexer scores
    iidx: u32,        // i32 [index_topk] selected positions (the gather idx; shared reuse target)
    ighist: u32,      // u32 [7*256] radix histograms (host-zeroed once)
    igctl: u32,       // u32 [3] grid-barrier ctl (host-zeroed once)
    icos: u32,
    isin: u32,
    // per-emitted-layer caches + weights (index i <-> layer_ids[i]); kidx = indexer key cache [ctx][DI]
    // on 'full' layers (TENSOR_NONE otherwise).
    ckv: Vec<u32>,
    krot: Vec<u32>,
    kidx: Vec<u32>,
    lw: Vec<GlmLW>,
}

/// Declare the GLM tensor set for the layers in `layer_ids` (real layer indices; the weight names
/// carry the real index so the prepped dir binds them). `lw[i]`/`ckv[i]`/`krot[i]` correspond to
/// `layer_ids[i]`. Activations are decode-shaped (one row).
fn declare_glm(b: &mut Builder, c: &GlmCfg, ctx: u32, layer_ids: &[u32]) -> GlmTn {
    let (h, nh, dk, dr, vd, ql, e, tk, imoe) = (
        c.hidden,
        c.heads,
        c.kv_lora,
        c.qk_rope,
        c.v_head,
        c.q_lora,
        c.n_exp,
        c.top_k,
        c.moe_inter,
    );
    // TENSOR-PARALLEL local shards (mirror the dense-GQA declare()): head-, expert- and
    // dense-intermediate-dimensioned tensors run 1/tp wide. tp==1 => *_l == full, byte-identical.
    let tp = c.tp;
    let nh_l = nh / tp; // this rank's q/v heads (column-parallel by head)
    let imoe_l = imoe / tp; // this rank's SHARED-expert/dense intermediate lanes (TP-sharded)
                            // Routed-expert intermediate width: full moe_inter under EP (whole experts, distributed across
                            // ranks — no CU-starve), else the TP shard. Sizes the `fu` gate/up buffer.
    let imoe_e = if c.ep { imoe } else { imoe_l };
    let di_l = c.dense_inter / tp; // this rank's dense-FFN intermediate lanes
    let ib = imoe.div_ceil(128); // expert scale-grid rows (I/128)
    let hb = h.div_ceil(128); // expert scale-grid cols (H/128)
    let db_l = di_l.div_ceil(128); // sharded dense scale-grid rows/cols (di_l/128)
    let db = c.dense_inter.div_ceil(128);
    let ac = |b: &mut Builder, n: &str, sz: u64| b.tensor(&format!("act.{n}"), sz);

    let ids = b.tensor("in.ids", ctx as u64 * I32);
    let pos = b.tensor("in.pos", ctx as u64 * I32);
    let kvlen = b.tensor("in.kvlen", I32);
    // Interleaved partial-RoPE cos/sin tables for the 64 rope dims (theta=8e6, full rotation of DR).
    // Same [ctx][DR/2] layout the half-split path uses (freq index = element>>1); the interp's HD=64
    // dispatch selects the INTERLEAVE=true template. See rope_tables + op_norm.h.
    let [cos_t, sin_t] = GenTensor::rope_pair(ctx, c.qk_rope, c.rope_theta, 1.0, RopeScale::None);
    let cos = b.tensor_gen("in.cos", cos_t.byte_len(), cos_t);
    let sin = b.tensor_gen("in.sin", sin_t.byte_len(), sin_t);
    let emb = b.tensor("model.embed_tokens.weight", (c.vocab * h) as u64 * BF16);
    let fin = b.tensor("model.norm.weight", h as u64 * BF16);
    let head = b.tensor("lm_head.weight", (c.vocab * h) as u64 * BF16);

    let x = ac(b, "x", h as u64 * BF16);
    let xn = ac(b, "xn", h as u64 * BF16);
    let qlr = ac(b, "qlr", ql as u64 * BF16);
    let qlat = ac(b, "qlat", ql as u64 * BF16);
    let ckvraw = ac(b, "ckvraw", dk as u64 * BF16);
    // Head-dimensioned activations shrink to nh_l heads under TP (the flash/merge/uv/o-fold ops run
    // this rank's head-shard); expert/dense-intermediate activations shrink to imoe_l/di_l lanes.
    let qa = ac(b, "qa", (nh_l * dk) as u64 * BF16);
    let qrr = ac(b, "qrr", (nh_l * dr) as u64 * BF16);
    let qr = ac(b, "qr", (nh_l * dr) as u64 * BF16);
    let krr = ac(b, "krr", dr as u64 * BF16);
    // TP-sharded head count (nh_l) x ctx-adaptive nsplit (glm_nsplit, from glm-tune-flash).
    // nh_l (not global c.heads) so the fill target matches this rank's actual work-item count.
    let ns = glm_nsplit(ctx, nh_l);
    let opart = ac(b, "opart", (nh_l * ns * dk) as u64 * F32);
    let mlpart = ac(b, "mlpart", (nh_l * ns * 2) as u64 * F32);
    let olat = ac(b, "olat", (nh_l * dk) as u64 * BF16);
    let oat = ac(b, "oat", (nh_l * vd) as u64 * BF16);
    let attn = ac(b, "attn", h as u64 * BF16);
    let xmid = ac(b, "xmid", h as u64 * BF16);
    let xn2 = ac(b, "xn2", h as u64 * BF16);
    let tab = ac(b, "tab", tk as u64 * 8);
    let rlogit = ac(b, "rlogit", e as u64 * BF16); // router score-GEMV output [n_exp] bf16
    let shfu = ac(b, "shfu", imoe_l as u64 * BF16);
    let shared = ac(b, "shared", h as u64 * BF16);
    // Routed-expert gate/up buffer: full moe_inter width per slot under EP (whole experts), else TP shard.
    let fu = ac(b, "fu", (tk * imoe_e) as u64 * BF16);
    let dfu = ac(b, "dfu", di_l as u64 * BF16);
    let part = ac(b, "part", (tk * h) as u64 * F32);
    let xnext = ac(b, "xnext", h as u64 * BF16);
    let logits = ac(b, "logits", c.vocab as u64 * BF16);
    let amax = ac(b, "amax.part", AMAX_BLOCKS as u64 * 8);
    // TP peer-mapped partials (§7a) — only under sharding; the host binds these into peer scratch at
    // offset 0 / slot_b so the row-parallel o_proj + MoE/dense down write peer-visible partials that
    // XReduce sums. zero_h is a persistent zero buffer used as the MoeCombine residual under TP (the
    // real residual xmid is added AFTER the all-reduce, so it is not summed N times).
    let og_tp = if tp > 1 {
        ac(b, "og_tp", h as u64 * BF16)
    } else {
        TENSOR_NONE
    };
    let dg_tp = if tp > 1 {
        ac(b, "dg_tp", h as u64 * BF16)
    } else {
        TENSOR_NONE
    };
    let zero_h = if tp > 1 {
        b.tensor_init("act.zero_h", vec![0u8; h as usize * 2])
    } else {
        TENSOR_NONE
    };

    // --- DSA lightning indexer scratch (ctx>2048 only). qidx/kidx/widx are per-step; iscore/iidx/
    //     ighist/igctl are the score+select scratch (shared across layers — decode runs them
    //     sequentially); icos/isin are the identity-tail interleaved-RoPE tables. ighist/igctl are
    //     tensor_init'd to ZERO (the coop select requires them clean on entry and leaves them clean). ---
    let dsa = c.dsa(ctx);
    let (hi, di, itk) = (c.index_heads, c.index_dim, c.index_topk.min(ctx));
    let (qidx, kidx_raw, kidx_normed, widx, iscore, iidx, ighist, igctl, icos, isin) = if dsa {
        let [ct, st] = GenTensor::rope_idx_pair(ctx, dr, di, c.rope_theta);
        (
            ac(b, "qidx", (hi * di) as u64 * BF16),
            ac(b, "kidx_raw", di as u64 * BF16),
            ac(b, "kidx_normed", di as u64 * BF16),
            ac(b, "widx", hi as u64 * BF16),
            ac(b, "iscore", ctx as u64 * F32),
            ac(b, "iidx", itk as u64 * I32),
            b.tensor_init("act.ighist", vec![0u8; 7 * 256 * 4]),
            b.tensor_init("act.igctl", vec![0u8; 3 * 4]),
            b.tensor_gen("in.icos", ct.byte_len(), ct),
            b.tensor_gen("in.isin", st.byte_len(), st),
        )
    } else {
        (
            TENSOR_NONE,
            TENSOR_NONE,
            TENSOR_NONE,
            TENSOR_NONE,
            TENSOR_NONE,
            TENSOR_NONE,
            TENSOR_NONE,
            TENSOR_NONE,
            TENSOR_NONE,
            TENSOR_NONE,
        )
    };

    let mut ckv = Vec::new();
    let mut krot = Vec::new();
    let mut kidx = Vec::new();
    let mut lw = Vec::new();
    for &l in layer_ids {
        ckv.push(b.tensor(&format!("kv.{l}.ckv"), (ctx * dk) as u64 * BF16));
        krot.push(b.tensor(&format!("kv.{l}.krot"), (ctx * dr) as u64 * BF16));
        // per-'full'-layer indexer key cache [ctx][DI] (accumulates like ckv/krot); shared layers none.
        let full = dsa && c.indexer_is_full(l);
        kidx.push(if full {
            b.tensor(&format!("kv.{l}.kidx"), (ctx * di) as u64 * BF16)
        } else {
            TENSOR_NONE
        });
        let t = |b: &mut Builder, s: &str, sz: u64| b.tensor(&format!("model.layers.{l}.{s}"), sz);
        let dense = c.is_dense(l);
        // The 256 per-expert block-fp8 weights + scale grids are NOT declared as .pkt tensors: the
        // loader binds them by name-pattern (model.layers.{l}.mlp.experts.{e}.{proj}.weight[_scale_inv])
        // straight from the prepped dir, packs them, and fills expert_weight_table/expert_scale_table.
        // Declaring 75*256*6 handles would bloat the tensor table for zero emit benefit (the MoE ops
        // index the tables, never the individual expert handles). ib/hb below size the scale grids.
        let _ = (ib, db);
        // Weight tensors carry this rank's SHARDED byte size (the host binds the matching slice):
        //   column-parallel (q/v absorb, q_rope, shared+dense+expert gate/up) -> nh_l/imoe_l/di_l rows;
        //   row-parallel (o_proj, shared+dense down) -> nh_l/imoe_l/di_l input lanes. tp==1 => full.
        //   Replicated (norms, q_a_proj, kv_a_latent, k_rope, router, bias) keep full dims.
        lw.push(GlmLW {
            gin: t(b, "input_layernorm.weight", h as u64 * BF16),
            qad: t(b, "self_attn.q_a_proj.weight", (ql * h) as u64 * BF16),
            gqa: t(b, "self_attn.q_a_layernorm.weight", ql as u64 * BF16),
            wqa: t(
                b,
                "self_attn.derived.q_absorb.weight",
                (nh_l * dk * ql) as u64 * BF16,
            ),
            wqr: t(
                b,
                "self_attn.derived.q_rope.weight",
                (nh_l * dr * ql) as u64 * BF16,
            ),
            ckvd: t(
                b,
                "self_attn.derived.kv_a_latent.weight",
                (dk * h) as u64 * BF16,
            ),
            gkva: t(b, "self_attn.kv_a_layernorm.weight", dk as u64 * BF16),
            krotd: t(b, "self_attn.derived.k_rope.weight", (dr * h) as u64 * BF16),
            wuv: t(
                b,
                "self_attn.derived.v_absorb.weight",
                (nh_l * dk * vd) as u64 * BF16,
            ),
            wo: t(b, "self_attn.o_proj.weight", (h * nh_l * vd) as u64 * BF16),
            gpost: t(b, "post_attention_layernorm.weight", h as u64 * BF16),
            wr: if dense {
                TENSOR_NONE
            } else {
                t(b, "mlp.gate.weight", (e * h) as u64 * BF16)
            },
            bias: if dense {
                TENSOR_NONE
            } else {
                t(b, "mlp.gate.e_score_correction_bias", e as u64 * F32)
            },
            shg: if dense {
                TENSOR_NONE
            } else {
                t(
                    b,
                    "mlp.shared_experts.gate_proj.weight",
                    (imoe_l * h) as u64 * BF16,
                )
            },
            shu: if dense {
                TENSOR_NONE
            } else {
                t(
                    b,
                    "mlp.shared_experts.up_proj.weight",
                    (imoe_l * h) as u64 * BF16,
                )
            },
            shd: if dense {
                TENSOR_NONE
            } else {
                t(
                    b,
                    "mlp.shared_experts.down_proj.weight",
                    (h * imoe_l) as u64 * BF16,
                )
            },
            ewt: if dense {
                TENSOR_NONE
            } else {
                t(b, "mlp.expert_weight_table", (e * 3) as u64 * 8)
            },
            est: if dense {
                TENSOR_NONE
            } else {
                t(b, "mlp.expert_scale_table", (e * 3) as u64 * 8)
            },
            dgate: if dense {
                t(b, "mlp.gate_proj.weight", (di_l * h) as u64)
            } else {
                TENSOR_NONE
            },
            dgate_s: if dense {
                t(
                    b,
                    "mlp.gate_proj.weight_scale_inv",
                    (db_l * hb) as u64 * F32,
                )
            } else {
                TENSOR_NONE
            },
            dup: if dense {
                t(b, "mlp.up_proj.weight", (di_l * h) as u64)
            } else {
                TENSOR_NONE
            },
            dup_s: if dense {
                t(b, "mlp.up_proj.weight_scale_inv", (db_l * hb) as u64 * F32)
            } else {
                TENSOR_NONE
            },
            ddown: if dense {
                t(b, "mlp.down_proj.weight", (h * di_l) as u64)
            } else {
                TENSOR_NONE
            },
            ddown_s: if dense {
                t(
                    b,
                    "mlp.down_proj.weight_scale_inv",
                    (hb * db_l) as u64 * F32,
                )
            } else {
                TENSOR_NONE
            },
            // DSA indexer weights (fp8 wq_b/wk copied VERBATIM for GemvFp8Blk + f32 [128,128] scale
            // grids; k_norm weight/bias + weights_proj bf16). REPLICATED across TP ranks (the indexer
            // is tiny and its idx is head-shared). Only bound on 'full' layers with the DSA gate on.
            iwqb: if full {
                t(b, "self_attn.indexer.wq_b.weight", (hi * di * ql) as u64)
            } else {
                TENSOR_NONE
            },
            iwqb_s: if full {
                t(
                    b,
                    "self_attn.indexer.wq_b.weight_scale_inv",
                    ((hi * di).div_ceil(128) * ql.div_ceil(128)) as u64 * F32,
                )
            } else {
                TENSOR_NONE
            },
            iwk: if full {
                t(b, "self_attn.indexer.wk.weight", (di * h) as u64)
            } else {
                TENSOR_NONE
            },
            iwk_s: if full {
                t(
                    b,
                    "self_attn.indexer.wk.weight_scale_inv",
                    (di.div_ceil(128) * hb) as u64 * F32,
                )
            } else {
                TENSOR_NONE
            },
            iknw: if full {
                t(b, "self_attn.indexer.k_norm.weight", di as u64 * BF16)
            } else {
                TENSOR_NONE
            },
            iknb: if full {
                t(b, "self_attn.indexer.k_norm.bias", di as u64 * BF16)
            } else {
                TENSOR_NONE
            },
            iwp: if full {
                t(
                    b,
                    "self_attn.indexer.weights_proj.weight",
                    (hi * h) as u64 * BF16,
                )
            } else {
                TENSOR_NONE
            },
        });
    }

    GlmTn {
        ids,
        pos,
        kvlen,
        cos,
        sin,
        emb,
        fin,
        head,
        x,
        xn,
        qlr,
        qlat,
        ckvraw,
        qa,
        qrr,
        qr,
        krr,
        opart,
        mlpart,
        olat,
        oat,
        attn,
        xmid,
        xn2,
        tab,
        rlogit,
        shfu,
        shared,
        fu,
        dfu,
        part,
        xnext,
        logits,
        amax,
        og_tp,
        dg_tp,
        zero_h,
        qidx,
        kidx_raw,
        kidx_normed,
        widx,
        iscore,
        iidx,
        ighist,
        igctl,
        icos,
        isin,
        ckv,
        krot,
        kidx,
        lw,
    }
}

/// Emit the shared MLA attention sub-block (input norm -> q/kv down + absorbed folds -> dynamic
/// interleaved RoPE on the 64 rope dims -> FLASH_MLA_DECODE -> merge -> O_UV_FOLD -> o_proj ->
/// residual -> post-attention norm). Writes `n.xn2` (the FFN input) and returns the post-attn-norm
/// completion dep. IDENTICAL for the dense (0-2) and MoE (3-77) layers, so both blocks call it.
fn emit_glm_mla(
    b: &mut Builder,
    c: &GlmCfg,
    n: &GlmTn,
    slot: usize,
    ctx: u32,
    x_in: u32,
    pre: &[u32],
    xgate: &mut u32,
    xr_cus: &[u32],
) -> u32 {
    let all = b.all();
    let one = vec![0u32];
    let (h, nh, dk, dr, vd, ql) = (c.hidden, c.heads, c.kv_lora, c.qk_rope, c.v_head, c.q_lora);
    let tp = c.tp;
    let nh_l = nh / tp; // this rank's head-shard (column-parallel by head); tp==1 => nh
    let w = &n.lw[slot];
    let eps = c.eps;
    // GEMV helper (M=1 decode, no norm fold) — the bf16 projection form both B4 passes used.
    let gemv = |b: &mut Builder, out: u32, x: u32, wt: u32, nn: u32, k: u32, deps: &[u32]| -> u32 {
        b.emit(DevOp::Gemv, all.clone(), deps, |d| {
            d.t[0] = out;
            d.t[1] = x;
            d.t[2] = wt;
            d.i[0] = 1;
            d.i[1] = nn;
            d.i[2] = k;
            d.f[0] = 1.0;
        })
    };
    // Standard Gemma-proven decode fusions (plans/glm52-fusion-audit.md). Each defaults ON; set the
    // env to "0" to emit the unfused baseline for a before/after measurement. A and G are byte-exact
    // (GemvQkv concatenates output columns — identical per-column dot/wave_sum/f2bf as the split
    // GEMVs); B1 is algebraically exact (AddNorm reduces over the un-rounded sum — see note below).
    let fuse_a = std::env::var("PLOW_GLM_FUSE_A").ok().as_deref() != Some("0");
    let fuse_g = std::env::var("PLOW_GLM_FUSE_G").ok().as_deref() != Some("0");
    // B1 defaults OFF (opt-in): AddNorm reduces over the un-rounded a+b sum, so unlike A/G it is NOT
    // byte-identical to the split Residual+RmsNorm — a reorder-level fp diff that flips one early
    // greedy argmax and cascades. Ship it only behind the HF-coherence gate; PLOW_GLM_FUSE_B1=1 opts in.
    let fuse_b1 = std::env::var("PLOW_GLM_FUSE_B1").ok().as_deref() == Some("1");

    // --- MLA ---
    // 1 input_layernorm
    // `pre` chains this layer's first op to the PREVIOUS layer's output (x_in), so the 78 layers run
    // in sequence rather than racing on the shared scratch/x buffers. Empty for the single-layer gate
    // (x_in is pre-uploaded before the launch, so no on-device producer to wait on).
    let c_rn1 = b.emit(DevOp::RmsNorm, one.clone(), pre, |d| {
        d.t[0] = n.xn;
        d.t[1] = x_in;
        d.t[2] = w.gin;
        d.i[0] = 1;
        d.i[1] = h;
        d.f[0] = eps;
    });
    // 2/6/8 down-projections. FUSION A (audit §A): q_a, kv_a and k_rope ALL read n.xn with K=h, so
    //   their output columns concatenate into ONE GemvQkv (Nq=ql q_a, Nk=dk kv_a, Nv=dr k_rope) that
    //   fills every wave (fixing the k_rope/kv_a CU-starvation) and deletes 2 gates/layer. Byte-exact
    //   to the three Gemvs. Legal: M*K = h fits GM_LDS_HALVES.
    let (c_qad, c_ckvd, c_krr) = if fuse_a {
        let c_fa = b.emit(DevOp::GemvQkv, all.clone(), &[c_rn1], |d| {
            d.t[0] = n.qlr;
            d.t[1] = n.xn;
            d.t[2] = w.qad; // q_a   -> Nq=ql
            d.t[3] = n.ckvraw;
            d.t[4] = w.ckvd; // kv_a  -> Nk=dk
            d.t[5] = n.krr;
            d.t[6] = w.krotd; // k_rope-> Nv=dr
            d.i[0] = 1;
            d.i[1] = ql;
            d.i[2] = h;
            d.i[3] = dk;
            d.i[4] = dr;
        });
        (c_fa, c_fa, c_fa)
    } else {
        (
            gemv(b, n.qlr, n.xn, w.qad, ql, h, &[c_rn1]),
            gemv(b, n.ckvraw, n.xn, w.ckvd, dk, h, &[c_rn1]),
            gemv(b, n.krr, n.xn, w.krotd, dr, h, &[c_rn1]),
        )
    };
    // 3 q_a_layernorm
    let c_rnq = b.emit(DevOp::RmsNorm, one.clone(), &[c_qad], |d| {
        d.t[0] = n.qlat;
        d.t[1] = n.qlr;
        d.t[2] = w.gqa;
        d.i[0] = 1;
        d.i[1] = ql;
        d.f[0] = eps;
    });
    // 4/5 absorbed q_nope (Wqa: QL -> NH_l*DK) and q_rope raw down (Wqr: QL -> NH_l*DR). FUSION G
    //   (audit §G): both read n.qlat with K=ql, so fuse into ONE GemvQkv with Nv=0 (q half + k half).
    //   Byte-exact. q_rope then gets a dynamic INTERLEAVED RoPE per head at pos (no norm); HD=64
    //   selects the interleaved template; q is not cached (out_row0/stride 0).
    let (c_qa, c_qrr) = if fuse_g {
        let c_fg = b.emit(DevOp::GemvQkv, all.clone(), &[c_rnq], |d| {
            d.t[0] = n.qa;
            d.t[1] = n.qlat;
            d.t[2] = w.wqa; // q_nope   -> Nq=nh_l*dk
            d.t[3] = n.qrr;
            d.t[4] = w.wqr; // q_rope raw-> Nk=nh_l*dr
            d.t[5] = TENSOR_NONE;
            d.t[6] = TENSOR_NONE; // Nv=0 (v branch never taken)
            d.i[0] = 1;
            d.i[1] = nh_l * dk;
            d.i[2] = ql;
            d.i[3] = nh_l * dr;
            d.i[4] = 0;
        });
        (c_fg, c_fg)
    } else {
        (
            gemv(b, n.qa, n.qlat, w.wqa, nh_l * dk, ql, &[c_rnq]),
            gemv(b, n.qrr, n.qlat, w.wqr, nh_l * dr, ql, &[c_rnq]),
        )
    };
    let c_qr = b.emit(DevOp::HeadNormRope, all.clone(), &[c_qrr], |d| {
        d.t[0] = n.qr;
        d.t[1] = n.qrr;
        d.t[2] = TENSOR_NONE;
        d.t[3] = n.cos;
        d.t[4] = n.sin;
        d.t[5] = n.pos;
        d.i[0] = 1;
        d.i[1] = nh_l;
        d.i[2] = dr;
        d.i[3] = 0;
        d.i[4] = 1;
        d.f[0] = eps;
        d.j[0] = 0;
        d.j[1] = KV_MASK_NONE;
    });
    // 7 kv_a_layernorm -> writes the latent cache (current row = row 0 here; the loader/decode
    //   step rebases the output to the current position, matching the ckv-row write of a decode step).
    //   Reads n.ckvraw from the fused (or split) down-projection above.
    let c_rnkv = b.emit(DevOp::RmsNorm, one.clone(), &[c_ckvd], |d| {
        d.t[0] = n.ckv[slot];
        d.t[1] = n.ckvraw;
        d.t[2] = w.gkva;
        d.i[0] = 1;
        d.i[1] = dk;
        d.f[0] = eps;
    });
    // 8 k_rope dynamic INTERLEAVED RoPE (shared 1-head) on n.krr from the fused (or split) down-proj,
    //   writing the rope cache at row=out_row0 (i[3]; the decode step patches it to the current pos).
    let c_krd = b.emit(DevOp::HeadNormRope, all.clone(), &[c_krr], |d| {
        d.t[0] = n.krot[slot];
        d.t[1] = n.krr;
        d.t[2] = TENSOR_NONE;
        d.t[3] = n.cos;
        d.t[4] = n.sin;
        d.t[5] = n.pos;
        d.i[0] = 1;
        d.i[1] = 1;
        d.i[2] = dr;
        d.i[3] = 0;
        d.i[4] = 1;
        d.f[0] = eps;
        d.j[0] = 0;
        d.j[1] = KV_MASK_NONE;
    });
    // --- DSA lightning indexer (G2/G5): ctx>2048 => project q_idx/k_idx/w, score, top-k select ->
    //     idx table, then FLASH_GATHER over the top_k selected latent rows. ctx<=2048 => dense flash
    //     (top-k is a no-op). 'full' layers own the indexer; 'shared' layers reuse the last full
    //     layer's idx (sequential layer chain => n.iidx already holds it). q_idx/k_idx use a HD=DI GPT-J
    //     interleaved RoPE with the identity-tail table (rope the first qk_rope=DR dims, pass the rest).
    let dsa = c.dsa(ctx);
    let full = dsa && w.iwqb != TENSOR_NONE; // 'full' indexer layer (weights bound only there)
    let itk = c.index_topk.min(ctx);
    let (hi, di) = (c.index_heads, c.index_dim);
    let gemv_blk = |b: &mut Builder,
                    out: u32,
                    x: u32,
                    wt: u32,
                    sc: u32,
                    nn: u32,
                    k: u32,
                    deps: &[u32]|
     -> u32 {
        b.emit(DevOp::GemvFp8Blk, all.clone(), deps, |d| {
            d.t[0] = out;
            d.t[1] = x;
            d.t[2] = wt;
            d.t[5] = sc;
            d.i[0] = 1;
            d.i[1] = nn;
            d.i[2] = k;
            d.i[4] = 0;
        })
    };
    let c_sel = if full {
        // q_idx = interleaved_rope(reshape_HIxDI(wq_b @ q_lat)); rope in-place (reads staged first).
        let c_q0 = gemv_blk(b, n.qidx, n.qlat, w.iwqb, w.iwqb_s, hi * di, ql, &[c_rnq]);
        let c_qi = b.emit(DevOp::HeadNormRope, all.clone(), &[c_q0], |d| {
            d.t[0] = n.qidx;
            d.t[1] = n.qidx;
            d.t[2] = TENSOR_NONE;
            d.t[3] = n.icos;
            d.t[4] = n.isin;
            d.t[5] = n.pos;
            d.i[0] = 1;
            d.i[1] = hi;
            d.i[2] = di;
            d.i[3] = 0;
            d.i[4] = 1;
            d.i[5] = 1;
            d.f[0] = eps;
            d.j[0] = 0;
            d.j[1] = KV_MASK_NONE;
        });
        // k_idx = interleaved_rope(k_norm_LAYERNORM+BIAS(wk @ xn)) cached [ctx][DI] at pos (like krot).
        let c_k0 = gemv_blk(b, n.kidx_raw, n.xn, w.iwk, w.iwk_s, di, h, &[c_rn1]);
        let c_kn = b.emit(DevOp::LayerNorm, one.clone(), &[c_k0], |d| {
            d.t[0] = n.kidx_normed;
            d.t[1] = n.kidx_raw;
            d.t[2] = w.iknw;
            d.t[3] = w.iknb;
            d.i[0] = 1;
            d.i[1] = di;
            d.i[3] = 0;
            d.f[0] = 1e-6; // k_norm eps
        });
        let c_ki = b.emit(DevOp::HeadNormRope, all.clone(), &[c_kn], |d| {
            d.t[0] = n.kidx[slot];
            d.t[1] = n.kidx_normed;
            d.t[2] = TENSOR_NONE;
            d.t[3] = n.icos;
            d.t[4] = n.isin;
            d.t[5] = n.pos;
            d.i[0] = 1;
            d.i[1] = 1;
            d.i[2] = di;
            d.i[3] = 0;
            d.i[4] = 1;
            d.i[5] = 1;
            d.f[0] = eps;
            d.j[0] = 0;
            d.j[1] = KV_MASK_NONE;
        });
        // w = weights_proj @ xn  [HI]  (bf16 GEMV)
        let c_w = gemv(b, n.widx, n.xn, w.iwp, hi, h, &[c_rn1]);
        // score[t] = Σ_h w[h]·ReLU(q_idx[h]·k_idx[t]) · scale  (scale = 1/√DI · 1/√HI; selection is
        // scale-invariant, this reproduces HF numerically).
        let c_sc = b.emit(DevOp::IndexScore, all.clone(), &[c_qi, c_ki, c_w], |d| {
            d.t[0] = n.iscore;
            d.t[1] = n.qidx;
            d.t[2] = n.kidx[slot];
            d.t[3] = n.widx;
            d.t[4] = n.kvlen;
            d.i[0] = 1;
            d.i[2] = ctx;
            d.f[0] = (di as f32).powf(-0.5) * (hi as f32).powf(-0.5);
        });
        // top-k SELECT -> n.iidx (ONE cooperative launch: grid-sync radix). Perf floor 2: emit on a
        // 32-CU slice, NOT all 256. The selector is grid-barrier CONTENTION-bound, not bandwidth-bound
        // (the score array is only ctx*4 B); cutting the co-resident WG count 256->32 drops the atomic
        // contention on the grid-sync counter and the shared histogram bins (~204->144us @128k, STILL
        // set-EXACT). The kernel reads nwg from in->blocks (=32) and grid-strides blockIdx.x over 0..31,
        // so CUs 0..31 give full, exact coverage; all 32 are trivially co-resident under the persistent
        // interp (256 CUs resident, this op gates on INDEX_SCORE, so its 32 WGs run together).
        let sel_wgs: Vec<u32> = (0..32.min(b.n_cu())).collect();
        b.emit(DevOp::IndexSelect, sel_wgs, &[c_sc], |d| {
            d.t[0] = n.iidx;
            d.t[1] = n.iscore;
            d.t[2] = n.ighist;
            d.t[3] = n.igctl;
            d.i[0] = ctx;
            d.i[1] = itk;
        })
    } else {
        0
    };
    // 9 FLASH (MLA) DECODE — dense (ctx<=2048) or GATHER over the top_k selected latent rows (ctx>2048).
    //   Runs this rank's nh_l head-shard; the latent ckv/krot caches are REPLICATED (all heads read
    //   the same shared latent), so the cache stays full-width on every rank. Under DSA the flash reads
    //   ONLY the top_k rows via n.iidx (constant work ~ top_k regardless of ctx).
    let ns_attn = if dsa {
        glm_nsplit(itk, nh_l)
    } else {
        glm_nsplit(ctx, nh_l)
    };
    let mut fl_deps = vec![c_qa, c_qr, c_rnkv, c_krd];
    if full {
        fl_deps.push(c_sel);
    }
    let c_fl = b.emit(
        if dsa {
            DevOp::FlashGatherDecode
        } else {
            DevOp::FlashMlaDecode
        },
        all.clone(),
        &fl_deps,
        |d| {
            d.t[0] = n.opart;
            d.t[1] = n.mlpart;
            d.t[2] = n.qa;
            d.t[3] = n.qr;
            d.t[4] = n.ckv[slot];
            d.t[5] = n.krot[slot];
            d.t[6] = n.kvlen;
            d.i[0] = 1;
            d.i[1] = nh_l;
            d.i[2] = ctx;
            d.i[4] = ns_attn;
            d.i[5] = KV_MASK_NONE;
            d.i[7] = glm_gf(ctx); // per-pkt head-fusion factor (interp dispatches GF=2/4 on this)
            d.f[0] = c.attn_scale;
            if dsa {
                d.t[7] = n.iidx; // idx table (this or the last full layer's selection)
                d.i[6] = itk; // top_k rows to gather
            }
        },
    );
    // 10 FUSED MLA MERGE+FOLD: online-softmax-merge the ns_attn latent partials (Opart/mlpart) in
    //    LDS, then fold olat @ W_uv straight to v_head_dim — replaces FLASH_MERGE<512> + O_UV_FOLD,
    //    killing the Olat[nh_l*DK] HBM round-trip and one dependency gate (validated rms ~0.004;
    //    ~1.1-1.24x on the MLA chain, composing with the ctx-scaled nsplit to 1.59x at 32k).
    let c_uv = b.emit(DevOp::MlaMergeFold, all.clone(), &[c_fl], |d| {
        d.t[0] = n.oat;
        d.t[1] = n.opart;
        d.t[2] = n.mlpart;
        d.t[3] = w.wuv;
        d.i[0] = 1;
        d.i[1] = nh_l;
        d.i[2] = vd;
        d.i[4] = ns_attn;
    });
    // 12 o_proj (NH_l*VD -> H)  [row-parallel]: each rank sums its head-shard into a PARTIAL H-vector.
    //   Under TP the partial goes to the peer-mapped og_tp slot and an XReduce all-reduces the N
    //   partials into n.attn; at tp==1 o_proj writes n.attn directly (byte-identical).
    // PLOW_NO_XREDUCE (diagnostic): drop the 156 all-reduce collectives (o_proj writes n.attn
    // directly with only this rank's partial) — numerically WRONG but same graph minus the
    // cross-GPU rendezvous, to isolate the XReduce cost. Never set for a real decode.
    let no_xr = tp > 1 && std::env::var("PLOW_NO_XREDUCE").ok().as_deref() == Some("1");
    let c_op = if tp > 1 && !no_xr {
        let c_p = gemv(b, n.og_tp, n.oat, w.wo, h, nh_l * vd, &[c_uv]);
        emit_xreduce(b, xgate, true, xr_cus, c_p, n.attn, h, tp, 0)
    } else {
        gemv(b, n.attn, n.oat, w.wo, h, nh_l * vd, &[c_uv])
    };
    // 13/14 post-attn residual + post_attention_layernorm. FUSION B1 (audit §B1): the plain add
    //   (xmid = x_in + attn) and the RmsNorm that re-reads it are the Qwen/Llama AddNorm pair — ONE
    //   packet writes BOTH the residual stream (xmid, consumed by the FFN combine) and its norm (xn2,
    //   the FFN input), deleting a gate/layer. NOTE: d_add_norm reduces over the UN-rounded a+b sum
    //   whereas the split path norms the bf16-rounded xmid, so this is algebraically exact but NOT
    //   guaranteed byte-identical to the split — the decode stream is verified before it is kept.
    if fuse_b1 {
        b.emit(DevOp::AddNorm, one.clone(), &[c_op], |d| {
            d.t[0] = n.xn2;
            d.t[1] = n.xmid;
            d.t[2] = x_in;
            d.t[3] = n.attn;
            d.t[4] = w.gpost;
            d.i[0] = 1;
            d.i[1] = h;
            d.f[0] = eps;
        })
    } else {
        let c_rs = b.emit(DevOp::Residual, one.clone(), &[c_op], |d| {
            d.t[0] = n.xmid;
            d.t[1] = x_in;
            d.t[2] = n.attn;
            d.i[0] = h;
            d.f[0] = 1.0;
        });
        b.emit(DevOp::RmsNorm, one.clone(), &[c_rs], |d| {
            d.t[0] = n.xn2;
            d.t[1] = n.xmid;
            d.t[2] = w.gpost;
            d.i[0] = 1;
            d.i[1] = h;
            d.f[0] = eps;
        })
    }
}

/// Emit ONE MoE (sparse) GLM decoder block — the exact block validated by the B4 harness. `slot`
/// indexes `tn.lw`/`tn.ckv`/`tn.krot`. `use_fp8` selects the block-fp8 expert opcodes (45/46) over
/// the bf16 ones (41/42). Returns the MoeCombine completion dep.
fn emit_glm_block(
    b: &mut Builder,
    c: &GlmCfg,
    n: &GlmTn,
    slot: usize,
    ctx: u32,
    use_fp8: bool,
    x_in: u32,
    x_out: u32,
    pre: &[u32],
    xgate: &mut u32,
    xr_cus: &[u32],
) -> u32 {
    assert!(slot < n.lw.len(), "slot out of range");
    let c_rn2 = emit_glm_mla(b, c, n, slot, ctx, x_in, pre, xgate, xr_cus);
    let all = b.all();
    let one = vec![0u32];
    let (h, e, tk, imoe) = (c.hidden, c.n_exp, c.top_k, c.moe_inter);
    let tp = c.tp;
    let imoe_l = imoe / tp; // this rank's SHARED-expert intermediate lanes (TP-sharded); tp==1 => imoe
                            // Routed-expert intermediate width: full moe_inter under EP (whole experts distributed across
                            // ranks — no CU-starve), else the TP shard. Under EP the host binds LOCAL experts (256/tp) whole,
                            // NULL for remote, and the kernel skips a null base; the combine XReduce folds the per-rank whole-
                            // expert partials in the same collective that already sums the shared partials.
    let imoe_e = if c.ep { imoe } else { imoe_l };
    let w = &n.lw[slot];
    let gemv = |b: &mut Builder, out: u32, x: u32, wt: u32, nn: u32, k: u32, deps: &[u32]| -> u32 {
        b.emit(DevOp::Gemv, all.clone(), deps, |d| {
            d.t[0] = out;
            d.t[1] = x;
            d.t[2] = wt;
            d.i[0] = 1;
            d.i[1] = nn;
            d.i[2] = k;
            d.f[0] = 1.0;
        })
    };
    // CONCURRENT EXPERT SEGMENTS (plans/glm52-coresident.md): the M=1 experts underfill 256 CUs
    // (latency-starved, ~12x above the weight-bandwidth roofline), so run the top_k chosen experts as
    // CO-RESIDENT segments — each owns a DISJOINT CU slice (tk experts x 256/tk CUs), all gated on the
    // SAME router counter, so all tk run at once instead of serially on all-256. Pure work-PARTITION
    // change (the kernel's slice/nblk mechanism does the rest): 0 = serial all-256 baseline, 1 =
    // concurrent experts (shared serial), 2 = concurrent experts + co-resident (proactive) shared expert.
    // SHIP DEFAULT = 1 (co-resident experts): bit-exact, measured -17.4% on the MoE block (the M=1
    // experts collapse from serial-all-256 to tk concurrent 256/tk-CU segments). GLM_MOE_CORESIDENT=0
    // restores the serial baseline; =2 adds the proactive co-resident shared expert (marginal, opt-in).
    let cores: u32 = std::env::var("GLM_MOE_CORESIDENT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    // Under cores>=2 the shared expert gets its own slice (parts = tk+1, slot tk), concurrent with the
    // tk routed experts; else it stays on all-256 (serial, ahead of the experts in the stream).
    let shared_cus = if cores >= 2 {
        b.split(tk + 1, tk)
    } else {
        all.clone()
    };
    // Per-slot routed-expert CU set: disjoint 1/tk (cores 1) or 1/(tk+1) (cores 2) slice, else all-256.
    let expert_parts = if cores >= 2 { tk + 1 } else { tk };

    // --- MoE ---
    // 15 router. DEFAULT (split): the 256-expert x K=6144 score matmul is the ordinary MULTI-CU
    //   wave-cooperative GEMV (all.clone()) — was the single-CU scalar dot that measured 73% of the
    //   MoE layer — feeding a cheap 1-CU MoeRouterTopk tail (bit-exact selection). GLM_ROUTER_OLD=1
    //   emits the fused single-CU d_moe_router for the before/after A/B.
    let c_router = if std::env::var("GLM_ROUTER_OLD").ok().as_deref() == Some("1") {
        b.emit(DevOp::MoeRouter, one.clone(), &[c_rn2], |d| {
            d.t[0] = n.tab;
            d.t[1] = n.xn2;
            d.t[2] = w.wr;
            d.t[3] = w.bias;
            d.i[0] = h;
            d.i[1] = e;
            d.i[2] = tk;
            d.i[3] = GLM_ROUTER_FLAGS;
            d.f[0] = c.route_scale;
        })
    } else {
        let c_score = gemv(b, n.rlogit, n.xn2, w.wr, e, h, &[c_rn2]);
        b.emit(DevOp::MoeRouterTopk, one.clone(), &[c_score], |d| {
            d.t[0] = n.tab;
            d.t[1] = n.rlogit;
            d.t[3] = w.bias;
            d.i[1] = e;
            d.i[2] = tk;
            d.i[3] = GLM_ROUTER_FLAGS;
            d.f[0] = c.route_scale;
        })
    };
    // 16 shared expert gate|up (fused GLU) — column-parallel: this rank's imoe_l lanes. Under cores>=2
    //   it runs on its OWN slice (shared_cus), CO-RESIDENT with the routed experts (it is routing-
    //   independent — gated only on c_rn2 — so it overlaps the expert chain instead of preceding it).
    let c_shglu = b.emit(DevOp::GemvGlu, shared_cus.clone(), &[c_rn2], |d| {
        d.t[0] = n.shfu;
        d.t[1] = n.xn2;
        d.t[2] = w.shg;
        d.t[5] = w.shu;
        d.i[0] = 1;
        d.i[1] = imoe_l;
        d.i[2] = h;
        d.i[5] = GLM_ACT_SILU;
    });
    // 17 shared expert down — row-parallel (imoe_l input): writes a PARTIAL H-vector under TP
    let c_shd = b.emit(DevOp::Gemv, shared_cus.clone(), &[c_shglu], |d| {
        d.t[0] = n.shared;
        d.t[1] = n.shfu;
        d.t[2] = w.shd;
        d.i[0] = 1;
        d.i[1] = h;
        d.i[2] = imoe_l;
        d.f[0] = 1.0;
    });
    // 18..33 the top-8 routed experts (gate/up GLU then down). imoe_e = full moe_inter under EP (whole
    //   experts, host binds the LOCAL 256/tp experts + NULL for remote; the kernel skips a null base),
    //   else the imoe_l TP shard. Each expert's part[slot] is an H-vector partial the combine XReduce
    //   folds. c.group collapses the 2*tk per-slot packets into 2 grouped packets (ops 48/49, fp8 only).
    let downs: Vec<u32> = if c.group && use_fp8 {
        // ONE grouped gate/up packet + ONE grouped down packet (op-count collapse for M=1 decode).
        let c_g = b.emit(DevOp::MoeGroupGluFp8Blk, all.clone(), &[c_router], |d| {
            d.t[0] = n.fu;
            d.t[1] = n.xn2;
            d.t[2] = n.tab;
            d.t[3] = w.ewt;
            d.t[4] = w.est;
            d.i[0] = tk;
            d.i[1] = imoe_e;
            d.i[2] = h;
            d.i[3] = e;
            d.i[5] = GLM_ACT_SILU;
        });
        let c_d = b.emit(DevOp::MoeGroupDownFp8Blk, all.clone(), &[c_g], |d| {
            d.t[0] = n.part;
            d.t[1] = n.fu;
            d.t[2] = n.tab;
            d.t[3] = w.ewt;
            d.t[4] = w.est;
            d.i[0] = tk;
            d.i[1] = h;
            d.i[2] = imoe_e;
            d.i[3] = e;
        });
        vec![c_d]
    } else {
        let (glu_op, down_op) = if use_fp8 {
            (DevOp::MoeExpertGluFp8Blk, DevOp::MoeExpertDownFp8Blk)
        } else {
            (DevOp::MoeExpertGlu, DevOp::MoeExpertDown)
        };
        let mut downs = Vec::with_capacity(tk as usize);
        for sl in 0..tk {
            // cores 0: all-256 (serial). cores>=1: disjoint 1/expert_parts slice → the tk experts
            //   (+ shared under cores 2) are co-resident and run concurrently, gated on c_router.
            let ecus = if cores >= 1 {
                b.split(expert_parts, sl)
            } else {
                all.clone()
            };
            let c_g = b.emit(glu_op, ecus.clone(), &[c_router], |d| {
                d.t[0] = n.fu;
                d.t[1] = n.xn2;
                d.t[2] = n.tab;
                d.t[3] = w.ewt;
                if use_fp8 {
                    d.t[4] = w.est;
                }
                d.i[0] = sl;
                d.i[1] = imoe_e;
                d.i[2] = h;
                d.i[3] = e;
                d.i[5] = GLM_ACT_SILU;
            });
            let c_d = b.emit(down_op, ecus, &[c_g], |d| {
                d.t[0] = n.part;
                d.t[1] = n.fu;
                d.t[2] = n.tab;
                d.t[3] = w.ewt;
                if use_fp8 {
                    d.t[4] = w.est;
                }
                d.i[0] = sl;
                d.i[1] = h;
                d.i[2] = imoe_e;
                d.i[3] = e;
            });
            downs.push(c_d);
        }
        downs
    };
    // 34 combine: sum shared + Σ gate·expert (f32 acc, fixed slot order). Under TP shared/part are
    //   PARTIALS, so the combine residual must NOT be xmid (it would be summed N times by XReduce);
    //   it writes the partial (residual = zero_h) into dg_tp, XReduce all-reduces into n.attn, and a
    //   Residual then adds the real xmid -> x_out. tp==1 keeps the fused xmid combine (byte-identical).
    let mut deps = Vec::with_capacity(1 + downs.len());
    deps.push(c_shd);
    deps.extend_from_slice(&downs);
    let no_xr = tp > 1 && std::env::var("PLOW_NO_XREDUCE").ok().as_deref() == Some("1");
    if tp > 1 && !no_xr {
        let c_cmb = b.emit(DevOp::MoeCombine, all.clone(), &deps, |d| {
            d.t[0] = n.dg_tp;
            d.t[1] = n.zero_h;
            d.t[2] = n.shared;
            d.t[3] = n.part;
            d.i[0] = h;
            d.i[1] = tk;
        });
        let slot_b = h * 2; // dg_tp peer offset (partial_A = og_tp @ 0, partial_B = dg_tp @ h*2)
        let c_xr = emit_xreduce(b, xgate, true, xr_cus, c_cmb, n.attn, h, tp, slot_b);
        b.emit(DevOp::Residual, one.clone(), &[c_xr], |d| {
            d.t[0] = x_out;
            d.t[1] = n.xmid;
            d.t[2] = n.attn;
            d.i[0] = h;
            d.f[0] = 1.0;
        })
    } else if tp > 1 && no_xr {
        // diagnostic: combine this rank's partials straight onto the residual, no all-reduce
        b.emit(DevOp::MoeCombine, all.clone(), &deps, |d| {
            d.t[0] = x_out;
            d.t[1] = n.xmid;
            d.t[2] = n.shared;
            d.t[3] = n.part;
            d.i[0] = h;
            d.i[1] = tk;
        })
    } else {
        b.emit(DevOp::MoeCombine, all.clone(), &deps, |d| {
            d.t[0] = x_out;
            d.t[1] = n.xmid;
            d.t[2] = n.shared;
            d.t[3] = n.part;
            d.i[0] = h;
            d.i[1] = tk;
        })
    }
}

/// Emit ONE DENSE (first_k_dense_replace) GLM decoder block — layers 0-2. The MLA attention is
/// identical to the MoE block; the FFN is a straight block-fp8 SwiGLU (no router/experts/shared):
/// DENSE_GLU_FP8_BLK (gate/up, H->dense_inter) -> GEMV_FP8_BLK (down, dense_inter->H) -> residual.
/// Returns the final residual completion dep (writes `n.xnext`).
fn emit_glm_dense_block(
    b: &mut Builder,
    c: &GlmCfg,
    n: &GlmTn,
    slot: usize,
    ctx: u32,
    x_in: u32,
    x_out: u32,
    pre: &[u32],
    xgate: &mut u32,
    xr_cus: &[u32],
) -> u32 {
    assert!(slot < n.lw.len(), "slot out of range");
    let c_rn2 = emit_glm_mla(b, c, n, slot, ctx, x_in, pre, xgate, xr_cus);
    let all = b.all();
    let one = vec![0u32];
    let (h, di) = (c.hidden, c.dense_inter);
    let tp = c.tp;
    let di_l = di / tp; // this rank's dense-FFN intermediate lanes; tp==1 => di
    let w = &n.lw[slot];
    // dense SwiGLU gate|up (block-fp8, op 47) — column-parallel: this rank's di_l lanes
    let c_glu = b.emit(DevOp::DenseGluFp8Blk, all.clone(), &[c_rn2], |d| {
        d.t[0] = n.dfu;
        d.t[1] = n.xn2;
        d.t[2] = w.dgate;
        d.t[5] = w.dup;
        d.t[3] = w.dgate_s;
        d.t[4] = w.dup_s;
        d.i[0] = di_l;
        d.i[1] = h;
        d.i[5] = GLM_ACT_SILU;
    });
    // dense down (block-fp8 GEMV, op 44) — row-parallel (di_l input). Under TP writes a PARTIAL into
    //   the dg_tp peer slot, XReduce all-reduces into n.attn, then residual; at tp==1 writes n.shared
    //   and the residual reads it directly (byte-identical).
    let no_xr = tp > 1 && std::env::var("PLOW_NO_XREDUCE").ok().as_deref() == Some("1");
    if tp > 1 && !no_xr {
        let c_down = b.emit(DevOp::GemvFp8Blk, all.clone(), &[c_glu], |d| {
            d.t[0] = n.dg_tp;
            d.t[1] = n.dfu;
            d.t[2] = w.ddown;
            d.t[5] = w.ddown_s;
            d.i[0] = 1;
            d.i[1] = h;
            d.i[2] = di_l;
            d.i[4] = 0;
        });
        let slot_b = h * 2;
        let c_xr = emit_xreduce(b, xgate, true, xr_cus, c_down, n.attn, h, tp, slot_b);
        b.emit(DevOp::Residual, one.clone(), &[c_xr], |d| {
            d.t[0] = x_out;
            d.t[1] = n.xmid;
            d.t[2] = n.attn;
            d.i[0] = h;
            d.f[0] = 1.0;
        })
    } else if tp > 1 && no_xr {
        let c_down = b.emit(DevOp::GemvFp8Blk, all.clone(), &[c_glu], |d| {
            d.t[0] = n.shared;
            d.t[1] = n.dfu;
            d.t[2] = w.ddown;
            d.t[5] = w.ddown_s;
            d.i[0] = 1;
            d.i[1] = h;
            d.i[2] = di_l;
            d.i[4] = 0;
        });
        b.emit(DevOp::Residual, one.clone(), &[c_down], |d| {
            d.t[0] = x_out;
            d.t[1] = n.xmid;
            d.t[2] = n.shared;
            d.i[0] = h;
            d.f[0] = 1.0;
        })
    } else {
        let c_down = b.emit(DevOp::GemvFp8Blk, all.clone(), &[c_glu], |d| {
            d.t[0] = n.shared;
            d.t[1] = n.dfu;
            d.t[2] = w.ddown;
            d.t[5] = w.ddown_s;
            d.i[0] = 1;
            d.i[1] = h;
            d.i[2] = di_l;
            d.i[4] = 0;
        });
        b.emit(DevOp::Residual, one.clone(), &[c_down], |d| {
            d.t[0] = x_out;
            d.t[1] = n.xmid;
            d.t[2] = n.shared;
            d.i[0] = h;
            d.f[0] = 1.0;
        })
    }
}

/// GLM-5.2 emit entry (Stack B: .pkt bound by name from the host-prepped weight dir). Milestone-1
/// emits the SINGLE-layer MoE block program the validation harness runs against the HF oracle; the
/// full 78-layer decode + dense layers + TP sharding are the next milestones.
/// Full 78-layer GLM-5.2 DECODE program (M=1): embed -> [dense 0-2 | MoE 3-77] ping-ponged -> final
/// norm -> lm_head -> argmax (writes the sampled id back into in.ids). Layers 0-77 (78 = MTP head,
/// skipped). Per-layer ckv/krot caches; the decode loop patches the current-token cache row per step
/// (k_rope HeadNormRope out_row0 via kv_row_insts; ckv RMSNORM output via a per-step pointer rebind).
/// `use_fp8` selects the block-fp8 expert kernels (45/46) for the MoE layers.
fn glm_emit_full(dir: &Path, ctx: u32, out: &str, n_cu: u32, tp: u32, use_fp8: bool, rope_gen: bool) {
    let mut c = cfg_glm(dir);
    c.tp = tp;
    // GLM_NLAYERS truncates the model to the first N layers — a single-GPU smoke test of the decode
    // LOOP mechanics (embed/chain/KV-row patch/argmax/multi-step) that fits without TP or all 78
    // layers' weights. Default = full 0..77 (layer 78 = MTP, skipped).
    let nl = std::env::var("GLM_NLAYERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(c.layers)
        .min(c.layers);
    let layers: Vec<u32> = (0..nl).collect();

    let mut tb = Builder::new(n_cu);
    let tn = declare_glm(&mut tb, &c, ctx, &layers);
    let tensors = tb.tensors();
    let gen = tb.gen_tensors();
    let mut b = Builder::new(n_cu);
    b.adopt_tensors(tensors.clone());
    let all = b.all();

    // embed: in.ids[0] -> x  (GLM has no embedding scale)
    let c_emb = b.emit(DevOp::Embed, all.clone(), &[], |d| {
        d.t[0] = tn.x;
        d.t[1] = tn.emb;
        d.t[2] = tn.ids;
        d.i[0] = 1;
        d.i[1] = c.hidden;
        d.f[0] = 1.0;
    });
    // 78 decoder layers, ping-ponging x <-> xnext so layer l+1 reads layer l's output. Each layer's
    // first op waits on the previous layer's completion (`dep`) — the layers run in sequence.
    // XReduce collectives (decode one-shot): each o_proj + FFN-down all-reduce takes a unique xctr
    // gate id (allocated by xgate). At tp==1 no XReduce is emitted. The all-reduce runs on `all` CUs
    // by default; PLOW_XR_CUS caps it (the TP8 NUMA-crossing lever, plans/tp-design.md §8b).
    let mut xgate: u32 = 0;
    let xr_cus: Vec<u32> = {
        let k = std::env::var("PLOW_XR_CUS")
            .ok()
            .and_then(|v| v.parse::<u32>().ok());
        match k {
            Some(k) if k > 0 && k < n_cu => (0..k).collect(),
            _ => all.clone(),
        }
    };
    let mut cur = tn.x;
    let mut dep = c_emb;
    for (slot, &l) in layers.iter().enumerate() {
        let nxt = if cur == tn.x { tn.xnext } else { tn.x };
        dep = if c.is_dense(l) {
            emit_glm_dense_block(
                &mut b,
                &c,
                &tn,
                slot,
                ctx,
                cur,
                nxt,
                &[dep],
                &mut xgate,
                &xr_cus,
            )
        } else {
            emit_glm_block(
                &mut b,
                &c,
                &tn,
                slot,
                ctx,
                use_fp8,
                cur,
                nxt,
                &[dep],
                &mut xgate,
                &xr_cus,
            )
        };
        cur = nxt;
    }
    // final RMSNorm (model.norm) -> xn, then lm_head GEMV -> logits, greedy argmax -> in.ids.
    let c_f = b.emit(DevOp::RmsNorm, vec![0u32], &[dep], |d| {
        d.t[0] = tn.xn;
        d.t[1] = cur;
        d.t[2] = tn.fin;
        d.i[0] = 1;
        d.i[1] = c.hidden;
        d.f[0] = c.eps;
    });
    let c_lm = b.emit(DevOp::Gemv, all.clone(), &[c_f], |d| {
        d.t[0] = tn.logits;
        d.t[1] = tn.xn;
        d.t[2] = tn.head;
        d.i[0] = 1;
        d.i[1] = c.vocab;
        d.i[2] = c.hidden;
        d.i[4] = 0;
    });
    let c_am = b.emit(DevOp::Argmax, (0..AMAX_BLOCKS).collect(), &[c_lm], |d| {
        d.t[0] = tn.amax;
        d.t[1] = tn.logits;
        d.i[0] = c.vocab;
    });
    b.emit(DevOp::ArgmaxFin, vec![0u32], &[c_am], |d| {
        d.t[0] = tn.ids;
        d.t[1] = tn.amax;
        d.i[0] = AMAX_BLOCKS;
    });
    let prog = b.finish();

    let n_ops = prog.insts.len();
    let mut m = Model {
        n_cu,
        tensors,
        progs: vec![prog],
        kv_row_insts: Vec::new(),
        prog_t: vec![1],
        gen,
    };
    if !rope_gen {
        m.bake_gen();
    }
    std::fs::write(out, m.to_blob()).unwrap();
    eprintln!(
        "glm52-FULL: {} layers (0-{}) hidden={} experts={}/top{} vocab={} {} -> {out}\n  \
         {n_ops} ops, decode M=1, ctx={ctx}, tp={tp}",
        layers.len(),
        layers.len().saturating_sub(1),
        c.hidden,
        c.n_exp,
        c.top_k,
        c.vocab,
        if use_fp8 { "block-fp8" } else { "bf16" }
    );
}

fn glm_main(dir: &Path, ctx: u32, out: &str, n_cu: u32, tp: u32, rope_gen: bool) {
    let use_fp8 = std::env::var("PLOW_FP8").ok().as_deref() == Some("1");
    // Full 78-layer serving decode program (GLM_FULL=1) vs the single-layer validation gate (default).
    if std::env::var("GLM_FULL").ok().as_deref() == Some("1") {
        glm_emit_full(dir, ctx, out, n_cu, tp, use_fp8, rope_gen);
        return;
    }
    let mut c = cfg_glm(dir);
    c.tp = tp;
    assert_eq!(
        tp, 1,
        "GLM TP sharding is milestone-3; use --tp 1 for the single-layer bring-up"
    );
    // Which layer to emit for the single-layer vs-HF gate (default = first MoE layer, matching the
    // B4 oracle's layer 3).
    let layer: u32 = std::env::var("GLM_LAYER")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(c.first_k_dense);
    let dense = c.is_dense(layer);

    let mut tb = Builder::new(n_cu);
    let tn = declare_glm(&mut tb, &c, ctx, &[layer]);
    let tensors = tb.tensors();
    let gen = tb.gen_tensors();
    let mut b = Builder::new(n_cu);
    b.adopt_tensors(tensors.clone());
    let mut xgate = 0u32; // tp==1 single-layer gate: no XReduce, so xgate/xr_cus are unused
    if dense {
        emit_glm_dense_block(
            &mut b,
            &c,
            &tn,
            0,
            ctx,
            tn.x,
            tn.xnext,
            &[],
            &mut xgate,
            &[],
        );
    } else {
        emit_glm_block(
            &mut b,
            &c,
            &tn,
            0,
            ctx,
            use_fp8,
            tn.x,
            tn.xnext,
            &[],
            &mut xgate,
            &[],
        );
    }
    let prog = b.finish();

    let n_ops = prog.insts.len();
    let mut m = Model {
        n_cu,
        tensors,
        progs: vec![prog],
        kv_row_insts: Vec::new(),
        prog_t: vec![1],
        gen,
    };
    if !rope_gen {
        m.bake_gen();
    }
    std::fs::write(out, m.to_blob()).unwrap();
    eprintln!(
        "glm52: GlmMoeDsa {} layers hidden={} heads={} kv_lora={} q_lora={} qk={}+{} v={} \
         experts={}/top{} moe_inter={} scale={:.4}",
        c.layers,
        c.hidden,
        c.heads,
        c.kv_lora,
        c.q_lora,
        c.qk_nope,
        c.qk_rope,
        c.v_head,
        c.n_exp,
        c.top_k,
        c.moe_inter,
        c.attn_scale
    );
    eprintln!(
        "  single-layer {} {} block: layer {layer}, {n_ops} ops, max_ctx={ctx} -> {out}",
        if dense {
            "block-fp8 DENSE"
        } else if use_fp8 {
            "block-fp8 MoE"
        } else {
            "bf16 MoE"
        },
        if dense {
            "SwiGLU (op47/44)"
        } else if use_fp8 {
            "MoeExpertGluFp8Blk/DownFp8Blk"
        } else {
            "MoeExpertGlu/Down"
        }
    );
    let _ = c.qk_head();
}

/// Parse a `--block` spec (`l` or `l..r`) into a bounds-checked half-open layer
/// range. Shared by the gemma and GLM emit paths.
fn parse_block(spec: &str, layers: usize) -> std::ops::Range<usize> {
    let r = if let Some((a, b)) = spec.split_once("..") {
        let lo: usize = a.trim().parse().expect("--block l..r: bad l");
        let hi: usize = b.trim().parse().expect("--block l..r: bad r");
        lo..hi
    } else {
        let l: usize = spec.trim().parse().expect("--block l: bad l");
        l..l + 1
    };
    assert!(
        r.start < r.end && r.end <= layers,
        "--block {r:?} out of range for a {layers}-layer model"
    );
    r
}

/// Serialize a block descriptor to pretty JSON, write a sibling `block.json` next to
/// `out`, and return the `SECT_METADATA` section that mirrors it into the blob.
/// Shared by the gemma and GLM `--block` emit paths.
fn write_block_descriptor(
    out: &str,
    desc: &plow_asset::BlockDescriptor,
) -> packet::devbuild::SectionData {
    let json = serde_json::to_vec_pretty(desc).expect("serialize block.json");
    let sib = std::path::Path::new(out).with_file_name("block.json");
    std::fs::write(&sib, &json).expect("write sibling block.json");
    packet::devbuild::SectionData {
        kind: packet::devbuild::SECT_METADATA,
        name: "block.json".into(),
        data: json,
    }
}

/// MLA+MoE emit flavor. The Model build (declare_glm + emit_glm_block/dense) is IDENTICAL across
/// these — only the descriptor's arch tag, mixer `kind`, and whether the DSA indexer role/dims/
/// carried indices apply differ. GLM-5.2 has the DSA lightning indexer; Kimi K2.7 / DeepSeek-V3 are
/// plain MLA (their cfg holds `has_dsa=false`, so the shared emit never takes the DSA path).
#[derive(Clone, Copy, PartialEq, Debug)]
enum MlaArch {
    Glm,
    Kimi,
    DeepSeek,
}

/// Build a single-block (layers `block`) MLA+MoE program + its descriptor, no file IO — the testable
/// core of `--block` on the GLM emit path (plans/block-asset-harness.md §5.3, §7) and, via `arch`,
/// the Kimi/DeepSeek reuse of that same emit (§5.0, M3). No embed / no final-norm+lm_head+argmax
/// tail: `act.x` in, the last layer's residual out. The emitter is slot-indexed (per-layer vectors
/// are built from `layer_ids`), so a range extraction is the existing single-layer bring-up
/// (glm_main default) generalized to N layers. `arch` selects only descriptor metadata — the ops
/// come from the shared GLM emit, DSA-gated on `c.dsa(ctx)` (held off for Kimi via cfg `has_dsa`).
fn glm_build_block(
    c: &GlmCfg,
    ctx: u32,
    n_cu: u32,
    block: std::ops::Range<usize>,
    use_fp8: bool,
    model: &str,
    arch: MlaArch,
) -> (Model, plow_asset::BlockDescriptor) {
    use plow_asset::*;
    let layers: Vec<u32> = block.clone().map(|l| l as u32).collect();

    let mut tb = Builder::new(n_cu);
    let tn = declare_glm(&mut tb, c, ctx, &layers);
    let tensors = tb.tensors();
    let gen = tb.gen_tensors();
    let mut b = Builder::new(n_cu);
    b.adopt_tensors(tensors.clone());
    let all = b.all();

    // Layer chain, ping-ponging x <-> xnext (layer l+1 reads layer l's output). The
    // first layer's first op has NO dependency (empty deps) — the block entry is
    // `act.x`, uploaded by the harness. tp==1 single-block: no XReduce, so xgate/xr_cus
    // are inert (mirrors glm_main's single-layer bring-up).
    let mut xgate = 0u32;
    let xr_cus = all.clone();
    let mut cur = tn.x;
    let mut dep: Vec<u32> = Vec::new();
    for (slot, &l) in layers.iter().enumerate() {
        let nxt = if cur == tn.x { tn.xnext } else { tn.x };
        let d = if c.is_dense(l) {
            emit_glm_dense_block(&mut b, c, &tn, slot, ctx, cur, nxt, &dep, &mut xgate, &xr_cus)
        } else {
            emit_glm_block(
                &mut b, c, &tn, slot, ctx, use_fp8, cur, nxt, &dep, &mut xgate, &xr_cus,
            )
        };
        dep = vec![d];
        cur = nxt;
    }
    // After N layers the residual is back in `x` (even) or in `xnext` (odd).
    let out_name = if cur == tn.x { "act.x" } else { "act.xnext" };
    let prog = b.finish();
    let mut m = Model {
        n_cu,
        tensors,
        progs: vec![prog],
        kv_row_insts: Vec::new(),
        prog_t: vec![1],
        gen,
    };

    // Descriptor. l0 = the extracted layer (block start); its DSA role + FFN kind
    // drive the arch-agnostic fields.
    let l0 = block.start as u32;
    let dsa_on = c.dsa(ctx);
    let full = c.indexer_is_full(l0);
    let dense = c.is_dense(l0);
    let hidden = c.hidden as i64;

    // MLA latent caches (ckv/krot) per layer; the indexer key cache (kidx) too on
    // 'full' indexer layers under an armed DSA gate.
    let mut kv_tensors = Vec::new();
    for l in block.clone() {
        kv_tensors.push(format!("kv.{l}.ckv"));
        kv_tensors.push(format!("kv.{l}.krot"));
        if dsa_on && c.indexer_is_full(l as u32) {
            kv_tensors.push(format!("kv.{l}.kidx"));
        }
    }
    let mut carried_state = vec![CarriedState {
        role: "kv".into(),
        tensors: kv_tensors,
        layout: "mla_latent".into(),
    }];
    // IndexShare (§7): a 'reuse' layer under an armed DSA gate consumes the previous
    // indexer layer's top-k selection — a carried INPUT, since the block does not
    // recompute it. (Gate off, or an 'indexer' layer => computed in-block => no carry.)
    if dsa_on && !full {
        carried_state.push(CarriedState {
            role: "dsa_indices".into(),
            tensors: vec!["act.iidx".into()],
            layout: "topk_positions".into(),
        });
    }

    // Arch-flavor metadata: GLM carries the DSA mixer kind + indexer role + index_* dims; Kimi/
    // DeepSeek are plain MLA (mla_attn, no dsa_role, no index_* dims). The ops are the same.
    let (arch_tag, mixer_kind) = match arch {
        MlaArch::Glm => ("glm_mla_dsa", "mla_dsa"),
        MlaArch::Kimi => ("kimi_mla_moe", "mla_attn"),
        MlaArch::DeepSeek => ("deepseek_mla_moe", "mla_attn"),
    };
    let is_glm = arch == MlaArch::Glm;
    let desc = BlockDescriptor {
        model: model.to_string(),
        arch: arch_tag.into(),
        layer: l0,
        kind: vec![
            mixer_kind.into(),
            if dense { "dense_ffn" } else { "moe_ffn" }.into(),
        ],
        hidden,
        dtype: if use_fp8 { "fp8".into() } else { "bf16".into() },
        dims: BlockDims {
            heads: Some(c.heads as i64),
            kv_lora: Some(c.kv_lora as i64),
            q_lora: Some(c.q_lora as i64),
            n_exp: (!dense).then_some(c.n_exp as i64),
            top_k: (!dense).then_some(c.top_k as i64),
            shared_exp: (!dense).then_some(1),
            moe_inter: (!dense).then_some(c.moe_inter as i64),
            index_heads: is_glm.then_some(c.index_heads as i64),
            index_dim: is_glm.then_some(c.index_dim as i64),
            index_topk: is_glm.then_some(c.index_topk as i64),
            ..Default::default()
        },
        // DSA role only on GLM; plain-MLA archs have no indexer (dsa_role absent).
        dsa_role: is_glm.then(|| if full { "indexer".into() } else { "reuse".into() }),
        inputs: vec![BlockTensor {
            name: "act.x".into(),
            shape: vec![Dim::Symbolic("T".into()), Dim::Fixed(hidden)],
            dtype: "bf16".into(),
        }],
        outputs: vec![BlockTensor {
            name: out_name.into(),
            shape: vec![Dim::Symbolic("T".into()), Dim::Fixed(hidden)],
            dtype: "bf16".into(),
        }],
        carried_state,
        weights: BlockWeights {
            mode: "symlink".into(),
            ckpt: model.to_string(),
            prefix: format!("model.layers.{l0}."),
        },
        programs: BlockPrograms {
            prefill_buckets: Vec::new(), // GLM emit path is decode-only (M=1)
            decode_t: 1,
        },
    };
    (m, desc)
}

/// `--block` on the GLM (glm_moe_dsa) emit path. Emits ONE block (layers `spec`) as a
/// GPU-loadable PLOWDEV blob with a `SECT_METADATA` `block.json` descriptor + sibling
/// file — the GLM analogue of the gemma `--block` path (decode-only; the GLM emitter
/// has no prefill program).
fn glm_emit_block(dir: &Path, ctx: u32, out: &str, n_cu: u32, tp: u32, spec: &str, rope_gen: bool) {
    let mut c = cfg_glm(dir);
    c.tp = tp;
    assert_eq!(
        tp, 1,
        "GLM TP sharding is milestone-3; use --tp 1 for --block extraction"
    );
    let use_fp8 = std::env::var("PLOW_FP8").ok().as_deref() == Some("1");
    let block = parse_block(spec, c.layers as usize);
    let model = dir
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| dir.to_string_lossy().into_owned());
    let (mut m, desc) = glm_build_block(&c, ctx, n_cu, block.clone(), use_fp8, &model, MlaArch::Glm);
    let section = write_block_descriptor(out, &desc);
    if !rope_gen {
        m.bake_gen();
    }
    std::fs::write(out, m.to_blob_v6(&[section])).unwrap();
    eprintln!(
        "glm52 --block {block:?}: {} block, {} layer(s), {} ops, dsa_role={} ctx={ctx} -> {out}",
        if use_fp8 { "block-fp8" } else { "bf16" },
        block.len(),
        m.progs[0].insts.len(),
        desc.dsa_role.as_deref().unwrap_or("-"),
    );
    eprintln!("  block.json sibling written next to {out}");
}

/// `--block` on the Kimi K2.7 / DeepSeek MLA+MoE path (plans/block-asset-harness.md §5.0/§5.3, M3).
/// Emits ONE block (layers `spec`) as a GPU-loadable PLOWDEV blob with a `SECT_METADATA` `block.json`
/// descriptor + sibling file. REUSES the GLM MLA + MoE emit verbatim (glm_build_block) with a Kimi
/// cfg (`has_dsa=false`) — no DSA, KV latent (ckv/krot) carried state, decode-only (the GLM emit has
/// no prefill program, so programs.prefill_buckets stays empty). `arch` picks the Kimi vs DeepSeek
/// descriptor tag.
fn kimi_emit_block(dir: &Path, ctx: u32, out: &str, n_cu: u32, tp: u32, spec: &str, arch: MlaArch, rope_gen: bool) {
    let mut c = cfg_kimi(dir);
    c.tp = tp;
    assert_eq!(
        tp, 1,
        "Kimi/DeepSeek TP sharding is a later milestone; use --tp 1 for --block extraction"
    );
    let use_fp8 = std::env::var("PLOW_FP8").ok().as_deref() == Some("1");
    let block = parse_block(spec, c.layers as usize);
    let model = dir
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| dir.to_string_lossy().into_owned());
    let (mut m, desc) = glm_build_block(&c, ctx, n_cu, block.clone(), use_fp8, &model, arch);
    let section = write_block_descriptor(out, &desc);
    if !rope_gen {
        m.bake_gen();
    }
    std::fs::write(out, m.to_blob_v6(&[section])).unwrap();
    eprintln!(
        "{} --block {block:?}: {} block, {} layer(s), {} ops, ctx={ctx} -> {out}",
        desc.arch,
        if use_fp8 { "block-fp8" } else { "bf16" },
        block.len(),
        m.progs[0].insts.len(),
    );
    eprintln!("  block.json sibling written next to {out}");
}

// ===== Nemotron-3 Mamba-2 hybrid (plans/block-asset-harness.md §7 Nemotron, §11 M4). =========
// Nemotron-3 Nano 30B-A3B is a HYBRID: 52 layers = 23 Mamba-2 mixers + 23 MoE FFNs + 6 GQA
// attentions, interleaved by a `hybrid_override_pattern` string. The Mamba-2 mixer is the
// genuinely NEW piece (the first state-space op in the tree — DevOp::Mamba2Scan, op_mamba.cuh,
// and the `mamba_ref` golden below). The GQA-attention and MoE layers REUSE existing DevOps
// (the same attn/MoE ops gemma/kimi emit), so only the mamba mixer is new work.

/// One Nemotron layer's role. The `hybrid_override_pattern` chars map: 'M' => Mamba-2 mixer,
/// '*' => GQA attention, '-' => MoE FFN (Nemotron-3 is the MoE variant, so the MLP slot is MoE).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum NemoKind {
    Mamba,
    Attn,
    Moe,
}

impl NemoKind {
    /// Descriptor `kind` tag for this layer.
    fn tag(self) -> &'static str {
        match self {
            NemoKind::Mamba => "mamba2",
            NemoKind::Attn => "gqa_attn",
            NemoKind::Moe => "moe_ffn",
        }
    }
}

/// Nemotron-3 hybrid config. Small synthetic values in tests; `cfg_nemotron` fills real dims from
/// `config.json`. Reference geometry (Nemotron-H / Nemotron-3 Nano 30B-A3B, assumption where a key
/// is absent): hidden 4096, mamba d_inner 8192 (expand 2), n_head 128, head_dim 64, d_state 128,
/// d_conv 4, n_groups 8; attn 32 heads / 8 kv-heads / head_dim 128; MoE 128 routed + 1 shared,
/// top_k 6, moe_inter 768.
struct NemoCfg {
    layers: usize,
    hidden: u32,
    // Mamba-2 mixer.
    d_inner: u32,
    n_head: u32,   // mamba_n_heads
    head_dim: u32, // d_inner / n_head
    d_state: u32,
    d_conv: u32,
    n_groups: u32,
    // GQA attention.
    attn_heads: u32,
    attn_kv_heads: u32,
    attn_head_dim: u32,
    // MoE.
    n_exp: u32,
    top_k: u32,
    shared_exp: u32,
    moe_inter: u32,
    eps: f32,
    kinds: Vec<NemoKind>,
}

impl NemoCfg {
    /// conv_dim = d_inner + 2*n_groups*d_state — the width the depthwise conv1d runs over (x,B,C).
    fn conv_dim(&self) -> u32 {
        self.d_inner + 2 * self.n_groups * self.d_state
    }
}

/// Parse a Nemotron-3 `config.json`. Where a key is absent (this box has no checkpoint), falls back
/// to the reference geometry above and NOTES it via the returned defaults. The per-layer pattern
/// comes from `hybrid_override_pattern` (M/*/- chars); absent, it synthesizes the documented
/// 23-mamba / 6-attn / 23-moe interleave.
fn cfg_nemotron(dir: &Path) -> NemoCfg {
    let v: Value =
        serde_json::from_slice(&std::fs::read(dir.join("config.json")).expect("config.json"))
            .unwrap();
    let gu = |k: &str, d: u32| v[k].as_u64().map(|x| x as u32).unwrap_or(d);
    let hidden = gu("hidden_size", 4096);
    let expand = gu("mamba_expand", 2);
    let d_inner = v["mamba_d_inner"]
        .as_u64()
        .map(|x| x as u32)
        .unwrap_or(expand * hidden);
    let n_head = gu("mamba_n_heads", 128);
    let head_dim = v["mamba_head_dim"]
        .as_u64()
        .map(|x| x as u32)
        .unwrap_or(d_inner / n_head.max(1));
    let layers = gu("num_hidden_layers", 52) as usize;
    // Per-layer kind pattern.
    let kinds: Vec<NemoKind> = match v["hybrid_override_pattern"].as_str() {
        Some(p) => p
            .chars()
            .filter(|c| matches!(c, 'M' | '*' | '-'))
            .map(|c| match c {
                'M' => NemoKind::Mamba,
                '*' => NemoKind::Attn,
                _ => NemoKind::Moe,
            })
            .collect(),
        // Assumption: no pattern in config -> the documented interleave. Attention every ~9th
        // layer (6 of 52), Mamba/MoE alternating otherwise.
        None => (0..layers)
            .map(|l| {
                if l % 9 == 4 {
                    NemoKind::Attn
                } else if l % 2 == 0 {
                    NemoKind::Mamba
                } else {
                    NemoKind::Moe
                }
            })
            .collect(),
    };
    NemoCfg {
        layers: kinds.len().max(layers),
        hidden,
        d_inner,
        n_head,
        head_dim,
        d_state: gu("mamba_d_state", 128),
        d_conv: gu("mamba_d_conv", 4),
        n_groups: gu("mamba_n_groups", 8),
        attn_heads: gu("num_attention_heads", 32),
        attn_kv_heads: gu("num_key_value_heads", 8),
        attn_head_dim: gu("attention_head_dim", 128),
        n_exp: gu("n_routed_experts", 128),
        top_k: gu("num_experts_per_tok", 6),
        shared_exp: gu("n_shared_experts", 1),
        moe_inter: gu("moe_intermediate_size", 768),
        eps: v["rms_norm_eps"].as_f64().map(|x| x as f32).unwrap_or(1e-5),
        kinds,
    }
}

/// Emit ONE Mamba-2 mixer layer (decode, M=1). RmsNorm -> 3 projection GEMVs (z / xBC / dt; these
/// are the column slices of the single in_proj) -> the new DevOp::Mamba2Scan mixer core (conv1d +
/// SSD scan + gated RMSNorm, reading/writing conv_state + ssm_state) -> out_proj GEMV -> residual.
/// Returns the residual counter (the block chain dep). `mamba.{l}.conv_state`/`ssm_state` are the
/// carried tensors the descriptor advertises.
fn emit_nemotron_mamba(
    b: &mut Builder,
    c: &NemoCfg,
    l: u32,
    cur: u32,
    nxt: u32,
    deps: &[u32],
) -> u32 {
    let bf = |b: &mut Builder, name: String, elems: u64| b.tensor(&name, elems * 2);
    let f32t = |b: &mut Builder, name: String, elems: u64| b.tensor(&name, elems * 4);
    let (h, di, nh, hd, ds, dc) = (
        c.hidden as u64,
        c.d_inner as u64,
        c.n_head as u64,
        c.head_dim as u64,
        c.d_state as u64,
        c.d_conv as u64,
    );
    let cd = c.conv_dim() as u64;
    let pfx = format!("mamba.{l}.");
    let cus = b.all();
    let one = vec![0u32];
    // input RMSNorm
    let xn = bf(b, format!("{pfx}xn"), di.max(h));
    let g_in = bf(b, format!("{pfx}norm_in.w"), h);
    let d_norm = b.emit(DevOp::RmsNorm, cus.clone(), deps, |i| {
        i.t[0] = xn;
        i.t[1] = cur;
        i.t[2] = g_in;
        i.i[0] = 1;
        i.i[1] = c.hidden;
        i.f[0] = c.eps;
    });
    // z / xBC / dt projections (in_proj column slices)
    let z = bf(b, format!("{pfx}z"), di);
    let wz = bf(b, format!("{pfx}z_proj.w"), di * h);
    let d_z = b.emit(DevOp::Gemv, cus.clone(), &[d_norm], |i| {
        i.t[0] = z;
        i.t[1] = xn;
        i.t[2] = wz;
        i.i[0] = 1;
        i.i[1] = c.d_inner;
        i.i[2] = c.hidden;
    });
    let xbc = bf(b, format!("{pfx}xbc"), cd);
    let wxbc = bf(b, format!("{pfx}xbc_proj.w"), cd * h);
    let d_xbc = b.emit(DevOp::Gemv, cus.clone(), &[d_norm], |i| {
        i.t[0] = xbc;
        i.t[1] = xn;
        i.t[2] = wxbc;
        i.i[0] = 1;
        i.i[1] = c.conv_dim();
        i.i[2] = c.hidden;
    });
    let dt = bf(b, format!("{pfx}dt"), nh);
    let wdt = bf(b, format!("{pfx}dt_proj.w"), nh * h);
    let d_dt = b.emit(DevOp::Gemv, cus.clone(), &[d_norm], |i| {
        i.t[0] = dt;
        i.t[1] = xn;
        i.t[2] = wdt;
        i.i[0] = 1;
        i.i[1] = c.n_head;
        i.i[2] = c.hidden;
    });
    // Mixer core (single-CU, correctness-first). Packed params: A_log|D|dt_bias|conv_b|norm_w.
    let mixed = bf(b, format!("{pfx}y"), di);
    let conv_w = bf(b, format!("{pfx}conv1d.w"), cd * dc);
    let params = f32t(b, format!("{pfx}ssm_params"), 3 * nh + cd + di);
    let conv_state = f32t(b, format!("{pfx}conv_state"), (dc - 1) * cd);
    let ssm_state = f32t(b, format!("{pfx}ssm_state"), nh * hd * ds);
    let d_scan = b.emit(DevOp::Mamba2Scan, one, &[d_z, d_xbc, d_dt], |i| {
        i.t[0] = mixed;
        i.t[1] = xbc;
        i.t[2] = dt;
        i.t[3] = z;
        i.t[4] = conv_w;
        i.t[5] = params;
        i.t[6] = conv_state;
        i.t[7] = ssm_state;
        i.i[0] = 1; // T (decode)
        i.i[1] = c.d_inner;
        i.i[2] = c.n_head;
        i.i[3] = c.head_dim;
        i.i[4] = c.d_state;
        i.i[5] = c.n_groups;
        i.i[6] = c.d_conv;
        i.i[7] = c.conv_dim();
        i.f[0] = c.eps;
    });
    // out_proj + residual.
    let op = bf(b, format!("{pfx}out"), h);
    let wout = bf(b, format!("{pfx}out_proj.w"), h * di);
    let d_out = b.emit(DevOp::Gemv, cus.clone(), &[d_scan], |i| {
        i.t[0] = op;
        i.t[1] = mixed;
        i.t[2] = wout;
        i.i[0] = 1;
        i.i[1] = c.hidden;
        i.i[2] = c.d_inner;
    });
    b.emit(DevOp::Residual, cus, &[d_out], |i| {
        i.t[0] = nxt;
        i.t[1] = cur;
        i.t[2] = op;
        i.i[0] = c.hidden;
        i.f[0] = 1.0;
    })
}

/// Emit ONE GQA attention layer (decode, M=1) reusing the existing attention DevOps
/// (the gemma/kimi decode path): RmsNorm -> fused GemvQkv -> HeadNormRope (q RoPE) ->
/// FlashDecode -> FlashMerge -> o_proj GEMV -> residual. KV cache (`kv.{l}.k/v`) is the
/// carried state.
fn emit_nemotron_attn(
    b: &mut Builder,
    c: &NemoCfg,
    l: u32,
    ctx: u32,
    cur: u32,
    nxt: u32,
    deps: &[u32],
) -> u32 {
    let bf = |b: &mut Builder, name: String, elems: u64| b.tensor(&name, elems * 2);
    let f32t = |b: &mut Builder, name: String, elems: u64| b.tensor(&name, elems * 4);
    let (h, nh, kvh, hd) = (
        c.hidden as u64,
        c.attn_heads as u64,
        c.attn_kv_heads as u64,
        c.attn_head_dim as u64,
    );
    let nq = nh * hd;
    let nkv = kvh * hd;
    let pfx = format!("attn.{l}.");
    let cus = b.all();
    let nsplit = 1u32;
    let xn = bf(b, format!("{pfx}xn"), h);
    let g_in = bf(b, format!("{pfx}norm_in.w"), h);
    let d_norm = b.emit(DevOp::RmsNorm, cus.clone(), deps, |i| {
        i.t[0] = xn;
        i.t[1] = cur;
        i.t[2] = g_in;
        i.i[0] = 1;
        i.i[1] = c.hidden;
        i.f[0] = c.eps;
    });
    let q = bf(b, format!("{pfx}q"), nq);
    let k = bf(b, format!("{pfx}kv.{l}.k"), (ctx as u64) * nkv);
    let vv = bf(b, format!("{pfx}kv.{l}.v"), (ctx as u64) * nkv);
    let wq = bf(b, format!("{pfx}q_proj.w"), nq * h);
    let wk = bf(b, format!("{pfx}k_proj.w"), nkv * h);
    let wv = bf(b, format!("{pfx}v_proj.w"), nkv * h);
    let d_qkv = b.emit(DevOp::GemvQkv, cus.clone(), &[d_norm], |i| {
        i.t[0] = q;
        i.t[1] = xn;
        i.t[2] = wq;
        i.t[3] = k;
        i.t[4] = wk;
        i.t[5] = vv;
        i.t[6] = wv;
        i.i[0] = 1;
        i.i[1] = nq as u32;
        i.i[2] = c.hidden;
        i.i[3] = nkv as u32;
        i.i[4] = nkv as u32;
    });
    let cos = bf(b, format!("{pfx}rope.cos"), (ctx as u64) * hd);
    let sin = bf(b, format!("{pfx}rope.sin"), (ctx as u64) * hd);
    let pos = b.tensor(&format!("{pfx}pos"), 4);
    let d_rope = b.emit(DevOp::HeadNormRope, cus.clone(), &[d_qkv], |i| {
        i.t[0] = q;
        i.t[1] = q;
        i.t[3] = cos;
        i.t[4] = sin;
        i.t[5] = pos;
        i.i[0] = 1;
        i.i[1] = c.attn_heads;
        i.i[2] = c.attn_head_dim;
    });
    let opart = f32t(b, format!("{pfx}opart"), nq * (nsplit as u64));
    let mlpart = f32t(b, format!("{pfx}mlpart"), nh * (nsplit as u64) * 2);
    let kvlen = b.tensor(&format!("{pfx}kvlen"), 4);
    let d_fd = b.emit(DevOp::FlashDecode, cus.clone(), &[d_rope], |i| {
        i.t[0] = opart;
        i.t[1] = mlpart;
        i.t[2] = q;
        i.t[3] = k;
        i.t[4] = vv;
        i.t[5] = kvlen;
        i.i[0] = 1;
        i.i[1] = c.attn_heads;
        i.i[2] = c.attn_kv_heads;
        i.i[3] = nkv as u32;
        i.i[5] = nsplit;
        i.i[6] = c.attn_head_dim;
        i.f[0] = (c.attn_head_dim as f32).powf(-0.5);
    });
    let ao = bf(b, format!("{pfx}ao"), nq);
    let d_merge = b.emit(DevOp::FlashMerge, cus.clone(), &[d_fd], |i| {
        i.t[0] = ao;
        i.t[1] = opart;
        i.t[2] = mlpart;
        i.i[0] = 1;
        i.i[1] = c.attn_heads;
        i.i[2] = nsplit;
        i.i[3] = c.attn_head_dim;
    });
    let op = bf(b, format!("{pfx}o"), h);
    let wo = bf(b, format!("{pfx}o_proj.w"), h * nq);
    let d_o = b.emit(DevOp::Gemv, cus.clone(), &[d_merge], |i| {
        i.t[0] = op;
        i.t[1] = ao;
        i.t[2] = wo;
        i.i[0] = 1;
        i.i[1] = c.hidden;
        i.i[2] = nq as u32;
    });
    b.emit(DevOp::Residual, cus, &[d_o], |i| {
        i.t[0] = nxt;
        i.t[1] = cur;
        i.t[2] = op;
        i.i[0] = c.hidden;
        i.f[0] = 1.0;
    })
}

/// Emit ONE MoE FFN layer (decode, M=1) reusing the existing MoE DevOps (the kimi/GLM MoE path):
/// RmsNorm -> router score GEMV -> MoeRouterTopk -> shared expert (GemvGlu + down GEMV) ->
/// top_k × (MoeExpertGlu, MoeExpertDown) -> MoeCombine. No carried state.
fn emit_nemotron_moe(
    b: &mut Builder,
    c: &NemoCfg,
    l: u32,
    cur: u32,
    nxt: u32,
    deps: &[u32],
) -> u32 {
    let bf = |b: &mut Builder, name: String, elems: u64| b.tensor(&name, elems * 2);
    let f32t = |b: &mut Builder, name: String, elems: u64| b.tensor(&name, elems * 4);
    let (h, ne, kk, im) = (
        c.hidden as u64,
        c.n_exp as u64,
        c.top_k as u64,
        c.moe_inter as u64,
    );
    let pfx = format!("moe.{l}.");
    let cus = b.all();
    let xn = bf(b, format!("{pfx}xn"), h);
    let g_in = bf(b, format!("{pfx}norm_in.w"), h);
    let d_norm = b.emit(DevOp::RmsNorm, cus.clone(), deps, |i| {
        i.t[0] = xn;
        i.t[1] = cur;
        i.t[2] = g_in;
        i.i[0] = 1;
        i.i[1] = c.hidden;
        i.f[0] = c.eps;
    });
    let logit = bf(b, format!("{pfx}logit"), ne);
    let wr = bf(b, format!("{pfx}router.w"), ne * h);
    let d_score = b.emit(DevOp::Gemv, cus.clone(), &[d_norm], |i| {
        i.t[0] = logit;
        i.t[1] = xn;
        i.t[2] = wr;
        i.i[0] = 1;
        i.i[1] = c.n_exp;
        i.i[2] = c.hidden;
    });
    let table = f32t(b, format!("{pfx}routing_table"), kk * 2);
    let d_topk = b.emit(DevOp::MoeRouterTopk, vec![0u32], &[d_score], |i| {
        i.t[0] = table;
        i.t[1] = logit;
        i.i[1] = c.n_exp;
        i.i[2] = c.top_k;
        i.f[0] = 1.0;
    });
    // Shared expert (always on).
    let sh_fu = bf(b, format!("{pfx}shared.fu"), im);
    let sh_gu = bf(b, format!("{pfx}shared.gate_up.w"), 2 * im * h);
    let d_sgu = b.emit(DevOp::GemvGlu, cus.clone(), &[d_norm], |i| {
        i.t[0] = sh_fu;
        i.t[1] = xn;
        i.t[2] = sh_gu;
        i.t[5] = sh_gu;
        i.i[0] = 1;
        i.i[1] = c.moe_inter;
        i.i[2] = c.hidden;
        i.i[5] = 1; // silu
    });
    let shared = bf(b, format!("{pfx}shared.out"), h);
    let sh_dn = bf(b, format!("{pfx}shared.down.w"), h * im);
    let d_sdn = b.emit(DevOp::Gemv, cus.clone(), &[d_sgu], |i| {
        i.t[0] = shared;
        i.t[1] = sh_fu;
        i.t[2] = sh_dn;
        i.i[0] = 1;
        i.i[1] = c.hidden;
        i.i[2] = c.moe_inter;
    });
    // Routed experts (per-slot, one glu+down each).
    let ewt = b.tensor(&format!("{pfx}expert_weight_table"), ne * 3 * 8);
    let fu = bf(b, format!("{pfx}fu"), kk * im);
    let part = f32t(b, format!("{pfx}part"), kk * h);
    let mut d_parts = Vec::new();
    for slot in 0..c.top_k {
        let d_glu = b.emit(DevOp::MoeExpertGlu, cus.clone(), &[d_topk], |i| {
            i.t[0] = fu;
            i.t[1] = xn;
            i.t[2] = table;
            i.t[3] = ewt;
            i.i[0] = slot;
            i.i[1] = c.moe_inter;
            i.i[2] = c.hidden;
            i.i[3] = c.n_exp;
            i.i[5] = 1;
        });
        let d_dn = b.emit(DevOp::MoeExpertDown, cus.clone(), &[d_glu], |i| {
            i.t[0] = part;
            i.t[1] = fu;
            i.t[2] = table;
            i.t[3] = ewt;
            i.i[0] = slot;
            i.i[1] = c.hidden;
            i.i[2] = c.moe_inter;
            i.i[3] = c.n_exp;
        });
        d_parts.push(d_dn);
    }
    let mut combine_deps = vec![d_sdn];
    combine_deps.extend(&d_parts);
    b.emit(DevOp::MoeCombine, cus, &combine_deps, |i| {
        i.t[0] = nxt;
        i.t[1] = cur;
        i.t[2] = shared;
        i.t[3] = part;
        i.i[0] = c.hidden;
        i.i[1] = c.top_k;
    })
}

/// Build a single-block (layers `block`) Nemotron-3 program + its descriptor, no file IO — the
/// testable core of `--block` on the nemotron_h path (§5.3, §7). Per the extracted layer's kind it
/// emits the NEW Mamba-2 mixer op, or reuses the GQA-attention emit, or reuses the MoE emit. No
/// embed / no final-norm+lm_head+argmax tail: `act.x` in, the last layer's residual out. Decode-only
/// (M=1), like the GLM/Kimi block path.
fn nemotron_build_block(
    c: &NemoCfg,
    ctx: u32,
    n_cu: u32,
    block: std::ops::Range<usize>,
    model: &str,
) -> (Model, plow_asset::BlockDescriptor) {
    use plow_asset::*;
    let mut b = Builder::new(n_cu);
    let x = b.tensor("act.x", (c.hidden as u64) * 2);
    let xnext = b.tensor("act.xnext", (c.hidden as u64) * 2);
    // Mandatory GpuEngine handles, zero-stubbed (mirrors declare_glm / the gemma
    // block path). The block emits no Embed / lm_head, so in.ids and act.logits
    // are inert; in.pos / in.kvlen are patched per decode step by the runtime.
    // Without these, GpuEngine::load rejects the blob ("missing in.ids/in.pos/
    // in.kvlen/act.logits") before any kernel launches.
    b.tensor("in.ids", ctx as u64 * 4);
    b.tensor("in.pos", ctx as u64 * 4);
    b.tensor("in.kvlen", 4); // batch = kvlen_bytes/4 = 1
    b.tensor("act.logits", 1024 * 2); // vocab stub (bf16); unused (no head)
    let mut cur = x;
    let mut dep: Vec<u32> = Vec::new();
    for &l in block.clone().collect::<Vec<_>>().iter() {
        let nxt = if cur == x { xnext } else { x };
        let kind = c.kinds[l];
        let d = match kind {
            NemoKind::Mamba => emit_nemotron_mamba(&mut b, c, l as u32, cur, nxt, &dep),
            NemoKind::Attn => emit_nemotron_attn(&mut b, c, l as u32, ctx, cur, nxt, &dep),
            NemoKind::Moe => emit_nemotron_moe(&mut b, c, l as u32, cur, nxt, &dep),
        };
        dep = vec![d];
        cur = nxt;
    }
    let out_name = if cur == x { "act.x" } else { "act.xnext" };
    let tensors = b.tensors();
    let gen = b.gen_tensors();
    let prog = b.finish();
    let mut m = Model {
        n_cu,
        tensors,
        progs: vec![prog],
        kv_row_insts: Vec::new(),
        prog_t: vec![1],
        gen,
    };

    // Descriptor. kind = per-layer tags; carried_state = union (conv+ssm per mamba, kv per attn,
    // none for moe); dims populated for whichever layer kinds appear in the block.
    let l0 = block.start as u32;
    let kinds: Vec<NemoKind> = block.clone().map(|l| c.kinds[l]).collect();
    let has_mamba = kinds.contains(&NemoKind::Mamba);
    let has_attn = kinds.contains(&NemoKind::Attn);
    let has_moe = kinds.contains(&NemoKind::Moe);
    let mut carried_state = Vec::new();
    for l in block.clone() {
        match c.kinds[l] {
            NemoKind::Mamba => {
                carried_state.push(CarriedState {
                    role: "conv".into(),
                    tensors: vec![format!("mamba.{l}.conv_state")],
                    layout: "conv".into(),
                });
                carried_state.push(CarriedState {
                    role: "ssm".into(),
                    tensors: vec![format!("mamba.{l}.ssm_state")],
                    layout: "ssm_head_major".into(),
                });
            }
            NemoKind::Attn => carried_state.push(CarriedState {
                role: "kv".into(),
                tensors: vec![format!("kv.{l}.k"), format!("kv.{l}.v")],
                layout: "head_major".into(),
            }),
            NemoKind::Moe => {}
        }
    }
    let hidden = c.hidden as i64;
    let desc = BlockDescriptor {
        model: model.to_string(),
        arch: "nemotron_h".into(),
        layer: l0,
        kind: kinds.iter().map(|k| k.tag().to_string()).collect(),
        hidden,
        dtype: "bf16".into(),
        dims: BlockDims {
            // Mamba-2.
            d_inner: has_mamba.then_some(c.d_inner as i64),
            n_head: has_mamba.then_some(c.n_head as i64),
            d_state: has_mamba.then_some(c.d_state as i64),
            d_conv: has_mamba.then_some(c.d_conv as i64),
            n_groups: has_mamba.then_some(c.n_groups as i64),
            // head_dim: mamba head width if a mamba layer, else attn head width.
            head_dim: if has_mamba {
                Some(c.head_dim as i64)
            } else if has_attn {
                Some(c.attn_head_dim as i64)
            } else {
                None
            },
            // GQA attention.
            heads: has_attn.then_some(c.attn_heads as i64),
            kv_heads: has_attn.then_some(c.attn_kv_heads as i64),
            // MoE.
            n_exp: has_moe.then_some(c.n_exp as i64),
            top_k: has_moe.then_some(c.top_k as i64),
            shared_exp: has_moe.then_some(c.shared_exp as i64),
            moe_inter: has_moe.then_some(c.moe_inter as i64),
            ..Default::default()
        },
        dsa_role: None,
        inputs: vec![BlockTensor {
            name: "act.x".into(),
            shape: vec![Dim::Symbolic("T".into()), Dim::Fixed(hidden)],
            dtype: "bf16".into(),
        }],
        outputs: vec![BlockTensor {
            name: out_name.into(),
            shape: vec![Dim::Symbolic("T".into()), Dim::Fixed(hidden)],
            dtype: "bf16".into(),
        }],
        carried_state,
        weights: BlockWeights {
            mode: "symlink".into(),
            ckpt: model.to_string(),
            prefix: format!("backbone.layers.{l0}."),
        },
        programs: BlockPrograms {
            prefill_buckets: Vec::new(), // decode-only (M=1) block emit
            decode_t: 1,
        },
    };
    (m, desc)
}

/// `--block` on the Nemotron-3 (nemotron_h) hybrid path (M4). Emits ONE block (layers `spec`) as a
/// GPU-loadable PLOWDEV blob with a `SECT_METADATA` `block.json` descriptor + sibling file. Per-layer
/// dispatch: the NEW Mamba-2 mixer op (mamba layers) or the reused GQA-attn / MoE emit.
fn nemotron_emit_block(dir: &Path, ctx: u32, out: &str, n_cu: u32, tp: u32, spec: &str, rope_gen: bool) {
    assert_eq!(tp, 1, "Nemotron TP sharding is a later milestone; use --tp 1 for --block");
    let c = cfg_nemotron(dir);
    let block = parse_block(spec, c.layers);
    let model = dir
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| dir.to_string_lossy().into_owned());
    let (mut m, desc) = nemotron_build_block(&c, ctx, n_cu, block.clone(), &model);
    let section = write_block_descriptor(out, &desc);
    if !rope_gen {
        m.bake_gen();
    }
    std::fs::write(out, m.to_blob_v6(&[section])).unwrap();
    eprintln!(
        "nemotron_h --block {block:?}: {} layer(s), {} ops, kinds={:?}, ctx={ctx} -> {out}",
        block.len(),
        m.progs[0].insts.len(),
        desc.kind,
    );
    eprintln!("  block.json sibling written next to {out}");
}

/// Everything the device-blob emitter needs from the caller. The `PLOW_*`
/// environment knobs (fp8, uniseg, decode-batch, …) are still read inside the
/// emit paths exactly as before; this struct carries only what used to be
/// positional/named CLI arguments so plowc and the legacy `gemma4` bin can both
/// drive the same code.
#[derive(Clone, Debug)]
pub struct EmitArgs {
    /// HuggingFace checkpoint directory (config.json + safetensors).
    pub dir: PathBuf,
    /// Max context tokens the program is compiled for.
    pub ctx: u32,
    /// Output `.pkt` path (a `block.json`/sidecar may be written next to it).
    pub out: String,
    /// Target executor (SM/CU) count.
    pub n_cu: u32,
    /// Tensor-parallel degree (>= 1).
    pub tp: u32,
    /// `--block l` or `l..r` (env `PLOW_BLOCK` fallback): single-block extract.
    pub block_spec: Option<String>,
    /// `--embed-cubin`: interpreter cubin embedded as a blob section.
    pub embed_cubin: Option<String>,
    /// `--embed-hsaco`: interpreter hsaco embedded as a blob section.
    pub embed_hsaco: Option<String>,
    /// Declare the RoPE tables as recipes the runtime materialises (v7 blob)
    /// instead of expanding them into the init section. On by default; the C
    /// harnesses under `runtime/tests/` need it off — see [`Model::bake_gen`].
    pub rope_gen: bool,
}

impl EmitArgs {
    /// Parse the legacy `gemma4`/`tinygemma` CLI: named flags anywhere, then
    /// positional `<model-dir> <max_ctx> <out.pkt> [n_cu]`. `PLOW_BLOCK` is the
    /// `--block` fallback. Preserved verbatim so the two entry points agree.
    pub fn from_cli(argv: impl Iterator<Item = String>) -> EmitArgs {
        let mut tp: u32 = 1;
        let mut embed_cubin: Option<String> = None;
        let mut embed_hsaco: Option<String> = None;
        let mut block_spec: Option<String> =
            std::env::var("PLOW_BLOCK").ok().filter(|s| !s.is_empty());
        let mut pos: Vec<String> = Vec::new();
        let mut it = argv;
        while let Some(a) = it.next() {
            match a.as_str() {
                "--tp" => {
                    tp = it
                        .next()
                        .expect("--tp needs a value")
                        .parse()
                        .expect("--tp N")
                }
                s if s.starts_with("--tp=") => tp = s[5..].parse().expect("--tp=N"),
                "--block" => {
                    block_spec = Some(it.next().expect("--block needs a value (l or l..r)"));
                }
                s if s.starts_with("--block=") => {
                    block_spec = Some(s["--block=".len()..].to_string());
                }
                "--embed-cubin" => {
                    embed_cubin = Some(it.next().expect("--embed-cubin needs a path"));
                }
                s if s.starts_with("--embed-cubin=") => {
                    embed_cubin = Some(s["--embed-cubin=".len()..].to_string());
                }
                "--embed-hsaco" => {
                    embed_hsaco = Some(it.next().expect("--embed-hsaco needs a path"));
                }
                s if s.starts_with("--embed-hsaco=") => {
                    embed_hsaco = Some(s["--embed-hsaco=".len()..].to_string());
                }
                _ => pos.push(a),
            }
        }
        let mut pa = pos.into_iter();
        let dir = PathBuf::from(
            pa.next()
                .expect("usage: gemma4 [--tp N] [--embed-cubin <path>] [--embed-hsaco <path>] <model-dir> <max_ctx> <out.pkt> [n_cu]"),
        );
        let ctx: u32 = pa.next().expect("max_ctx").parse().unwrap();
        let out = pa.next().unwrap_or_else(|| "gemma4.pkt".into());
        let n_cu: u32 = pa.next().and_then(|s| s.parse().ok()).unwrap_or(256);
        EmitArgs { dir, ctx, out, n_cu, tp, block_spec, embed_cubin, embed_hsaco, rope_gen: true }
    }
}

/// Compile a checkpoint into a PLOWDEV device blob at `args.out`. This is the
/// former `gemma4` binary's `main`, verbatim below the argument parsing — the
/// same arch dispatch, the same env knobs, the same byte output.
pub fn run(args: EmitArgs) {
    let EmitArgs { dir, ctx, out, n_cu, tp, block_spec, embed_cubin, embed_hsaco, rope_gen } = args;

    // GLM-5.2 (GlmMoeDsa) — MLA + DSA + block-fp8 MoE — is a wholly separate emit path (glm_main).
    // Dispatch on model_type before the dense-GQA cfg parse, which would panic on GLM's config.
    let model_type =
        serde_json::from_slice::<Value>(&std::fs::read(dir.join("config.json")).unwrap())
            .ok()
            .and_then(|v| {
                v.get("model_type")
                    .and_then(|m| m.as_str())
                    .map(str::to_string)
            })
            .unwrap_or_default();
    if model_type == "glm_moe_dsa" {
        // GLM `--block` (M2, plans/block-asset-harness.md §5.3/§7): single-block
        // extraction on the separate GLM emitter. Absent => the unchanged glm_main
        // path (byte-identical).
        match &block_spec {
            Some(spec) => glm_emit_block(&dir, ctx, &out, n_cu, tp, spec, rope_gen),
            None => glm_main(&dir, ctx, &out, n_cu, tp, rope_gen),
        }
        return;
    }
    // Kimi K2.7 / DeepSeek-V2/V3 (plan §5.0/§5.3, M3): plain MLA + MoE, reusing GLM's MLA + MoE emit
    // (NOT rewrite/kimi.rs). model_type "kimi_k2"/"kimi" => Kimi tag; "deepseek_v3"/"deepseek_v2" =>
    // DeepSeek tag. Only the block-extraction (`--block`) device path is wired in M3; a full-model
    // Kimi device emit (the glm_main analogue) is a later milestone.
    if matches!(
        model_type.as_str(),
        "kimi_k2" | "kimi" | "deepseek_v3" | "deepseek_v2"
    ) {
        let arch = if model_type.starts_with("kimi") {
            MlaArch::Kimi
        } else {
            MlaArch::DeepSeek
        };
        match &block_spec {
            Some(spec) => kimi_emit_block(&dir, ctx, &out, n_cu, tp, spec, arch, rope_gen),
            None => panic!(
                "{model_type}: M3 supports only single-block extraction on the device path — pass \
                 --block <l>[..<r>] (or PLOW_BLOCK). Full-model Kimi/DeepSeek device emit is a later \
                 milestone (plans/block-asset-harness.md §5.3)."
            ),
        }
        return;
    }
    // Nemotron-3 Mamba-2 hybrid (plan §7, M4): mamba mixer (NEW op) + GQA attn + MoE, one block at a
    // time. Only the `--block` device path is wired in M4; a full-model Nemotron emit is a later
    // milestone (the hybrid layer-count + carried-state plumbing).
    if matches!(model_type.as_str(), "nemotron_h" | "nemotron3" | "nemotron") {
        match &block_spec {
            Some(spec) => nemotron_emit_block(&dir, ctx, &out, n_cu, tp, spec, rope_gen),
            None => panic!(
                "{model_type}: M4 supports only single-block extraction — pass --block <l>[..<r>] \
                 (or PLOW_BLOCK). Full-model Nemotron device emit is a later milestone \
                 (plans/block-asset-harness.md §5.3/§7)."
            ),
        }
        return;
    }

    let mut c = cfg_from(&dir);
    assert!(tp >= 1, "--tp must be >= 1");
    c.tp = tp;
    // Resolve the block range now that layer count is known. `l` -> l..l+1;
    // `l..r` -> that half-open range. Absent => the full model (0..layers),
    // which makes every gated site below byte-identical to the pre-block path.
    let block: std::ops::Range<usize> = match &block_spec {
        None => 0..c.layers as usize,
        Some(s) => parse_block(s, c.layers as usize),
    };
    let block_mode = block_spec.is_some();
    // FP8 (PLOW_FP8=1). The 7 projections gain an fp8 (w8a16) twin + per-channel scale. DECODE emits
    // GEMV_FP8 / GEMV_GLU_FP8; PREFILL emits GEMM_FP8 / GEMM_GLU_FP8 (T6 L2 — dequant-to-bf16-in-smem
    // + existing bf16 mma, per-channel scale in the epilogue). Both phases consume the fp8 twins, so
    // the bf16 projection weights are elided in fp8 mode (see `wproj`). The bf16 pkt is byte-identical
    // when unset. See runtime/nvidia/op_gemm.cuh, runtime/amd/op_gemm.h and gemma4_chat.c.
    let fp8 = std::env::var("PLOW_FP8").ok().as_deref() == Some("1");
    // T8 w8a8 (PLOW_W8A8=1, requires PLOW_FP8=1). PREFILL emits the true fp8 tensor-core path:
    // ONE per-row DevOp::QuantFp8 per activation site + GEMM_FP8/GEMM_GLU_FP8 re-pointed at the
    // fp8 activation (t1=xq) + a_scale (t3). The SAME opcodes serve T6 w8a16 (bf16 activation) —
    // the interp cubin selects the kernel by PLOW_NV_W8A8, so the w8a8 pkt MUST run against a
    // PLOW_NV_W8A8=1 prefill cubin (the T6 cubin would misread xq bytes as bf16). Weight side =
    // the same e4m3 twins + per-channel scales T6 declared. Unset => byte-identical emission.
    let w8a8 = std::env::var("PLOW_W8A8").ok().as_deref() == Some("1");
    assert!(
        !w8a8 || fp8,
        "PLOW_W8A8=1 requires PLOW_FP8=1 (the fp8 weight twins + scales)"
    );
    // FP8 KV-CACHE (PLOW_FP8_KV=1). Stores/reads K/V as e4m3 with a per-row f32 scale, halving the
    // decode KV stream (the HBM-bound part of flash-decode) and the KV footprint. Independent of the
    // fp8 WEIGHT path above so both can be A/B'd; the harness routes an fp8-KV pkt to the _fp8kv
    // interpreter objects (which carry the fp8 flash + HeadNormRopeFp8 arms).
    let fp8_kv = std::env::var("PLOW_FP8_KV").ok().as_deref() == Some("1");
    // PLOW_FP8_KV_FULL=1: restrict the e4m3 cache to FULL-attention (hd512) layers — the shape
    // the beat-fp8-mma PIPE=1 fp8-mma prefill flash serves. Requires PLOW_FP8_KV=1.
    let fp8_kv_full = fp8_kv && std::env::var("PLOW_FP8_KV_FULL").ok().as_deref() == Some("1");
    // layer_scalar is a Gemma-only learned per-layer residual scale; Llama/Qwen fold nothing here.
    let ls = if c.arch == Arch::Gemma4 {
        layer_scalars(&dir, c.layers, &c.prefix)
    } else {
        vec![1.0f32; c.layers as usize]
    };

    // Prefill BUCKETS. A 20-token prompt must not pay for a 4096-token program, and T is a
    // compile-time constant of the packets — so the compiler emits several and the runtime
    // picks the smallest that fits. This is what a shape bucket IS.
    // CAPPED AT MAX_CHUNK. Chunked prefill never emits a chunk larger than PLOW_MAX_CHUNK, so a
    // program for T > MAX_CHUNK can never be invoked -- the ladder used to run to 131072 and
    // every rung above 4096 was dead code that still cost compile time and packet size.
    // Tensor-parallel now emits SHARDED PREFILL buckets too (plans/tp-prefill.md): every prefill
    // op is Megatron-sharded in emit_phase (q/k/v/gate/up column-parallel, o/down row-parallel with
    // an XReduce all-reduce, flash head-split) exactly as decode is — the [T,hidden] all-reduce is
    // the only new regime. The full ladder is emitted at every tp; tp==1 stays byte-identical.
    let buckets: Vec<u32> = [128u32, 512, 1024, 2048, 4096, 8192]
        .into_iter()
        .filter(|&x| x <= ctx.min(MAX_CHUNK))
        .collect();
    // The invariant that ties MAX_CHUNK to KV_RING (see dev_isa.h). Break it and a chunk's own
    // rows wrap onto their history: a silent wrong answer, not a crash.
    assert!(
        KV_RING >= c.window + MAX_CHUNK - 1,
        "KV ring {KV_RING} too small for window {} + chunk {MAX_CHUNK}",
        c.window
    );
    let arows = ctx.min(MAX_CHUNK);
    // opart/mlpart (the flash_prefill partials) are sized in declare() as arows*heads_sharded*ns_pre.
    // The flash writes t*heads_sharded*ns(t) row-splits for a bucket t, where emit_phase derives
    // ns(t) from the SHARDED head count (heads/tp) — so ns_pre must be the worst-case over buckets
    // using that same sharded count, or a high-tp small-bucket program overflows opart (at tp=8 the
    // real ns is 32x the unsharded estimate → a GPU write fault). tp==1: hs==heads, ns_pre==1 (the
    // old value, byte-identical). plans/tp-prefill.md.
    let hs = (c.heads / c.tp).max(1);
    let max_splits = buckets
        .iter()
        .map(|&t| {
            let ns = n_cu.div_ceil((t.div_ceil(Q_TILE_ROWS) * hs).max(1)).max(1);
            t * hs * ns
        })
        .max()
        .unwrap_or(n_cu * Q_TILE_ROWS);
    let ns_pre = max_splits.div_ceil((arows * hs).max(1)).max(1);

    // BATCH>1 DECODE (serving pending #4, "max users supported"): PLOW_DECODE_BATCH=B emits a
    // WORKING batch-B decode program — KV cache, activations, GEMV M, flash n_batch, per-sequence
    // argmax all sized/set for B. B=1 (default) is byte-identical to the pre-batch blob (the serving
    // engine depends on it). B is capped at 32 — serving up to 32 concurrent users. The GEMV
    // ladder instantiates MM in {1,2,4,8} and every dispatcher (d_gemv, d_gemv_qkv, d_gemv_glu
    // and the fp8 twins) walks M in blocks of 8 above that, so the cap is a policy/KV-footprint
    // choice, not a kernel limit: the KV cache is sized dbatch* (7 GiB/seq at ctx=132k on 12B),
    // so B=32 only fits at a reduced ctx. Raising it further needs no kernel work.
    let dbatch: u32 = std::env::var("PLOW_DECODE_BATCH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1)
        .clamp(1, 32);
    // 26B-A4B MoE decode is BATCHED (B in 1..=32): the router family, the flat expert GLU/down
    // and the combine all carry a batch row count and index [B][k] routing slots. See the
    // work-item ordering note in runtime/nvidia/op_moe.cuh for the weight-reuse design.
    // (The fp8 batch refusal is gone: the fp8 GEMV arms are batched as of the B=32 work.)
    assert!(
        !(c.moe && dbatch > 32),
        "MoE decode batch is capped at 32 (per-CTA inv[] scratch, PLOW_MOE_MAXB)"
    );

    // Grouped-MoE PREFILL (plans/p9-26b-prefill-moe.md): token-sorted grouped expert GEMM buckets.
    // Enabled by default for the 26B-A4B MoE bf16 path; PLOW_MOE_PREFILL=0 restores the decode-only
    // blob (byte-identical to the pre-prefill build — the buffer sizing and new tensors are gated on
    // this flag). beat26b: fp8 grouped MoE prefill is now implemented for the w8a8 path (ops 81/82),
    // so it is enabled under PLOW_W8A8; plain fp8 (w8a16 dequant) grouped prefill is still not
    // implemented and stays decode-only.
    let moe_pf = c.moe
        && (!fp8 || w8a8)
        && std::env::var("PLOW_MOE_PREFILL").ok().as_deref() != Some("0");

    let mut tb = Builder::new(n_cu);
    let tn = declare(
        &mut tb, &c, ctx, ns_pre, fp8, w8a8, fp8_kv, fp8_kv_full, dbatch, moe_pf, block.clone(),
    );
    let tensors = tb.tensors();
    let gen = tb.gen_tensors();

    let mut progs = Vec::new();
    let mut tlist = Vec::new();
    for &t in &buckets {
        if c.moe && !moe_pf {
            break;
        } // MoE without prefill: decode-only blob
        let mut b = Builder::new(n_cu);
        b.adopt_tensors(tensors.clone());
        let mut dummy = Vec::new();
        emit_phase(
            &mut b,
            &c,
            &ls,
            &tn,
            t,
            ctx,
            Mode::Prefill,
            n_cu,
            &mut dummy,
            fp8,
            w8a8,
            fp8_kv,
            fp8_kv_full,
            block.clone(),
            block_mode,
        );
        progs.push(b.finish());
        tlist.push(t);
    }
    let mut bd = Builder::new(n_cu);
    bd.adopt_tensors(tensors.clone());
    let mut kv_rows = Vec::new();
    // `dbatch` is the SAME clamped(1,32) value used by declare() above — emission and
    // allocation must agree, so we reuse it here rather than re-reading the env (an unclamped
    // re-read would emit B>32 ops against buffers declare() sized for 32 → OOB writes).
    // DECODE-TILED (PLOW_DECODE_TILED=1, plans/decode-tiled.md): emit the decode bucket from
    // PREFILL kernels — tiled GEMM + FlashPrefill at one query row — instead of the GEMV family.
    // Targets long context, where GEMV does not scale with batch and FlashDecode caps at n_cu.
    // Unset emits a byte-identical program. **The sm_120 interpreter traps on every prefill
    // opcode** (interp_sm120.cu default arm), so this is AMD-only until those kernels exist; it
    // is a loud trap, not silent garbage. Correctness bar is a token stream IDENTICAL to the
    // Mode::Decode bucket at the same prompt — not "it ran".
    let dmode = if std::env::var("PLOW_DECODE_TILED").ok().as_deref() == Some("1") {
        Mode::DecodeTiled
    } else {
        Mode::Decode
    };
    emit_phase(
        &mut bd,
        &c,
        &ls,
        &tn,
        dbatch,
        ctx,
        dmode,
        n_cu,
        &mut kv_rows,
        fp8,
        false,
        fp8_kv,
        fp8_kv_full,
        block.clone(),
        block_mode,
    );
    progs.push(bd.finish());
    tlist.push(dbatch);

    let mut m = Model {
        n_cu,
        tensors,
        progs,
        kv_row_insts: kv_rows,
        prog_t: tlist,
        gen,
    };

    // Emit v6 with sections when --embed-cubin/--embed-hsaco given, else v5.
    let mut sections = Vec::new();
    // BLOCK MODE: embed the block.json descriptor (plans/block-asset-harness.md
    // §4) as SECT_METADATA — this also forces the to_blob_v6 path — and drop a
    // sibling block.json next to the blob for the record / the harness loader.
    if block_mode {
        use plow_asset::*;
        let l0 = block.start;
        let full = c.is_full[l0];
        let head_dim = if full { c.hd_full } else { c.hd_slide };
        let kv_heads = if full { c.kvh_full } else { c.kvh_slide };
        let kv_tensors: Vec<String> = block
            .clone()
            .flat_map(|l| [format!("kv.{l}.k"), format!("kv.{l}.v")])
            .collect();
        let hidden = c.hidden as i64;
        let ckpt = dir
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| dir.to_string_lossy().into_owned());
        // Top-level dtype reflects the actual compile (fp8 weight twins vs
        // bf16); the act.x tensors stay bf16 (fp8 is weight-only w8a16).
        let desc = BlockDescriptor {
            model: ckpt.clone(),
            arch: "gemma_dense".into(),
            layer: l0 as u32,
            kind: vec!["dense_attn".into(), "dense_ffn".into()],
            hidden,
            dtype: if fp8 { "fp8".into() } else { "bf16".into() },
            dims: BlockDims {
                heads: Some(c.heads as i64),
                head_dim: Some(head_dim as i64),
                kv_heads: Some(kv_heads as i64),
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
                tensors: kv_tensors,
                layout: "head_major".into(),
            }],
            weights: BlockWeights {
                mode: "symlink".into(),
                ckpt,
                prefix: format!("{}layers.{}.", c.prefix, l0),
            },
            programs: BlockPrograms {
                prefill_buckets: buckets.iter().map(|&t| t as i64).collect(),
                decode_t: dbatch as i64,
            },
        };
        sections.push(write_block_descriptor(&out, &desc));
        eprintln!("  block mode: layers {block:?}");
    }
    if let Some(ref path) = embed_cubin {
        sections.push(packet::devbuild::SectionData {
            kind: packet::devbuild::SECT_CUBIN,
            name: "interp_sm120".into(),
            data: std::fs::read(path).expect("--embed-cubin: cannot read file"),
        });
    }
    if let Some(ref path) = embed_hsaco {
        sections.push(packet::devbuild::SectionData {
            kind: packet::devbuild::SECT_HSACO,
            name: "interp_gfx950".into(),
            data: std::fs::read(path).expect("--embed-hsaco: cannot read file"),
        });
    }
    if !rope_gen {
        m.bake_gen();
    }
    let blob = if sections.is_empty() {
        m.to_blob()
    } else {
        m.to_blob_v6(&sections)
    };
    // Coverage gate BEFORE the blob lands on disk: a wrong .pkt that exists will be
    // benchmarked by someone. PLOW_SKIP_COVERAGE=1 is the deliberate escape hatch for
    // partial/renamed checkpoints; it is loud because it re-arms the silent-wrong-model
    // failure mode this gate exists to prevent.
    match validate_coverage(&dir, &c.prefix, &m.tensors.iter().map(|t| t.name.clone()).collect::<Vec<_>>()) {
        Ok(()) => {}
        Err(e) if std::env::var("PLOW_SKIP_COVERAGE").ok().as_deref() == Some("1") => {
            eprintln!("*** PLOW_SKIP_COVERAGE=1 — EMITTING A MODEL KNOWN TO BE WRONG ***\n{e}");
        }
        Err(e) => {
            eprintln!("gemma4: {e}");
            std::process::exit(1);
        }
    }
    std::fs::write(&out, blob).unwrap();

    let wb: u64 = m
        .tensors
        .iter()
        .filter(|x| x.name.starts_with("model.") || x.name.starts_with("fp8/"))
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
        "gemma4: {} layers ({} full)  hidden={} inter={}  heads={}  hd={}/{}  kvh={}/{}  vocab={}",
        c.layers,
        c.is_full.iter().filter(|x| **x).count(),
        c.hidden,
        c.inter,
        c.heads,
        c.hd_slide,
        c.hd_full,
        c.kvh_slide,
        c.kvh_full,
        c.vocab
    );
    eprintln!("  max_ctx={}  prefill buckets {:?} + decode", ctx, buckets);
    eprintln!("  layer_scalar[0..4] = {:?}", &ls[..4.min(ls.len())]);
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


fn split2(n: u32, a: u32, b: u32) -> (Vec<u32>, Vec<u32>) {
    let s = (((n as u64 * a as u64) / (a + b).max(1) as u64).max(1) as u32).min(n - 1);
    ((0..s).collect(), (s..n).collect())
}
fn split3(n: u32, a: u32, b: u32, c: u32) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    if c == 0 {
        let (x, y) = split2(n, a, b);
        return (x, y, Vec::new());
    }
    let tot = (a + b + c).max(1) as u64;
    let sa = (((n as u64 * a as u64) / tot).max(1) as u32).min(n - 2);
    let sb = (((n as u64 * b as u64) / tot).max(1) as u32).min(n - sa - 1);
    (
        (0..sa).collect(),
        (sa..sa + sb).collect(),
        (sa + sb..n).collect(),
    )
}

#[cfg(test)]
mod glm_tests {
    //! The GLM-5.2 (GlmMoeDsa) single-layer emit is the FIRST milestone-1 gate: the emitted op
    //! sequence must be identical to the 34-op MoE block that runtime/tests/
    //! glm52_real_block_gfx950_test.c validated on gfx950 against the HF oracle (real 256 experts,
    //! real [128,128] block-fp8 scales — plans/glm52-campaign.md "B4-CORE DONE"). Asserting op-for-op
    //! equality here, offline, means the emitted layer inherits that passing GPU result. No GPU, no
    //! weights — a pure structural equivalence proof, exactly as the Gemma pick_tile tests lock in
    //! the tile choice offline.
    use super::*;

    /// The real GLM-5.2-FP8 config dims (plans/glm52-arch.md). `layers` is trimmed — the single
    /// block only touches one layer.
    fn glm_ref_cfg() -> GlmCfg {
        GlmCfg {
            layers: 4,
            hidden: 6144,
            heads: 64,
            kv_lora: 512,
            q_lora: 2048,
            qk_nope: 192,
            qk_rope: 64,
            v_head: 256,
            vocab: 154880,
            eps: 1e-5,
            n_exp: 256,
            top_k: 8,
            moe_inter: 2048,
            dense_inter: 12288,
            first_k_dense: 3,
            route_scale: 2.5,
            attn_scale: (256f32).powf(-0.5),
            rope_theta: 8_000_000.0,
            tp: 1,
            ep: false,
            group: false,
            index_heads: 32,
            index_dim: 128,
            index_topk: 2048,
            // indexer_types[0..4] = full,full,full,shared (real GLM-5.2 pattern); irrelevant to these
            // ctx=512 offline tests (DSA is gated OFF at ctx<=2048) but set for completeness.
            indexer_full: vec![true, true, true, false],
            has_dsa: true,
        }
    }

    fn emitted_ops(use_fp8: bool) -> Vec<u16> {
        let c = glm_ref_cfg();
        let mut b = Builder::new(256);
        // Emit MoE layer 3 (the B4 oracle's layer), matching the harness.
        let tn = declare_glm(&mut b, &c, 512, &[3]);
        let tensors = b.tensors();
        let mut b2 = Builder::new(256);
        b2.adopt_tensors(tensors);
        let mut xgate = 0u32;
        emit_glm_block(
            &mut b2,
            &c,
            &tn,
            0,
            512,
            use_fp8,
            tn.x,
            tn.xnext,
            &[],
            &mut xgate,
            &[],
        );
        b2.finish().insts.iter().map(|d| d.op).collect()
    }

    /// The reference MoE-block op sequence, in emission order. This is the B4 harness sequence
    /// (glm52_real_block_gfx950_test.c) with the two rope-slice GEMVs each followed by a dynamic
    /// interleaved HeadNormRope (HD=64) instead of a position-FOLDED GEMV — the production form that
    /// runtime/tests/glm52_run.c validates on gfx950 (dynamic rope at a fixed position reproduces the
    /// folded B4 numbers). The folded B4 result is inherited by transitivity: dynamic-at-fixed-pos ==
    /// the fold, proven numerically by the glm52_run ms1 gate.
    fn ref_sequence(use_fp8: bool) -> Vec<u16> {
        use DevOp::*;
        let (glu, down) = if use_fp8 {
            (MoeExpertGluFp8Blk, MoeExpertDownFp8Blk)
        } else {
            (MoeExpertGlu, MoeExpertDown)
        };
        let mut ops = vec![
            RmsNorm,        // input_layernorm
            GemvQkv, // FUSED A: q_a + kv_a + k_rope input projections (share xn) -> one GemvQkv
            RmsNorm, // q_a_layernorm
            GemvQkv, // FUSED G: Wqa (absorbed q_nope) + Wqr (q_rope) -> one GemvQkv
            HeadNormRope, // q_rope dynamic interleaved RoPE (HD=64)
            RmsNorm, // kv_a_layernorm -> latent cache
            HeadNormRope, // k_rope dynamic interleaved RoPE -> rope cache
            FlashMlaDecode, // MLA flash
            MlaMergeFold, // fused latent merge + W_uv fold (was FlashMerge + OUvFold)
            Gemv,    // o_proj
            Residual, // x_mid
            RmsNorm, // post_attention_layernorm
            Gemv,    // router SCORE GEMV (multi-CU wave-cooperative; the router split)
            MoeRouterTopk, // router tail: sigmoid+bias+norm_topk+scale (1-CU bit-exact selection)
            GemvGlu, // shared expert gate|up
            Gemv,    // shared expert down
        ];
        for _ in 0..8 {
            ops.push(glu);
            ops.push(down);
        }
        ops.push(MoeCombine);
        ops.into_iter().map(|o| o as u16).collect()
    }

    #[test]
    fn glm_block_matches_reference_bf16() {
        assert_eq!(
            emitted_ops(false),
            ref_sequence(false),
            "bf16 op sequence != reference"
        );
    }

    #[test]
    fn glm_block_matches_reference_fp8() {
        assert_eq!(
            emitted_ops(true),
            ref_sequence(true),
            "block-fp8 op sequence != reference"
        );
    }

    /// The dense (layers 0-2) block op sequence: shared MLA (16 ops) + block-fp8 SwiGLU (dense GLU
    /// op 47, dense down GEMV_FP8_BLK op 44) + residual = 19 ops.
    fn emitted_dense_ops() -> Vec<u16> {
        let c = glm_ref_cfg();
        let mut b = Builder::new(256);
        let tn = declare_glm(&mut b, &c, 512, &[0]); // layer 0 is dense (first_k_dense_replace=3)
        let tensors = b.tensors();
        let mut b2 = Builder::new(256);
        b2.adopt_tensors(tensors);
        let mut xgate = 0u32;
        emit_glm_dense_block(
            &mut b2,
            &c,
            &tn,
            0,
            512,
            tn.x,
            tn.xnext,
            &[],
            &mut xgate,
            &[],
        );
        b2.finish().insts.iter().map(|d| d.op).collect()
    }

    /// Emit ONE MoE layer (slot layer 3) at `ctx`, with the indexer 'full'/'shared'/off, and return
    /// the op sequence. `full` binds an indexer on layer 3; `ctx>2048` arms the DSA gate.
    fn emitted_ops_dsa(ctx: u32, full: bool) -> Vec<u16> {
        let mut c = glm_ref_cfg();
        c.indexer_full = vec![false, false, false, full]; // layer 3 = MoE; full toggles its indexer
        let mut b = Builder::new(256);
        let tn = declare_glm(&mut b, &c, ctx, &[3]);
        let tensors = b.tensors();
        let mut b2 = Builder::new(256);
        b2.adopt_tensors(tensors);
        let mut xgate = 0u32;
        emit_glm_block(
            &mut b2,
            &c,
            &tn,
            0,
            ctx,
            true,
            tn.x,
            tn.xnext,
            &[],
            &mut xgate,
            &[],
        );
        b2.finish().insts.iter().map(|d| d.op).collect()
    }

    #[test]
    fn glm_dsa_gate_off_below_cutover() {
        use DevOp::*;
        // ctx<=CROSSOVER (65536): NO DSA ops, dense FlashMlaDecode — byte-identical to the non-DSA MoE
        // block. 32768 is in the mid-ctx band, where the measured full-model TP4 winner is dense.
        let ops = emitted_ops_dsa(32768, true);
        assert!(
            ops.contains(&(FlashMlaDecode as u16)),
            "dense flash below cutover"
        );
        assert!(
            !ops.contains(&(FlashGatherDecode as u16)),
            "no gather below cutover"
        );
        assert!(
            !ops.contains(&(IndexScore as u16)),
            "no indexer below cutover"
        );
        assert_eq!(
            ops,
            ref_sequence(true),
            "ctx<=2048 == plain MoE block (DSA off)"
        );
    }

    #[test]
    fn glm_dsa_full_layer_emits_indexer() {
        use DevOp::*;
        // ctx>CROSSOVER, 'full': indexer (2 fp8 projections + LayerNorm + 2 rope + weights_proj GEMV +
        // score + select) then FLASH_GATHER (not dense).
        let ops = emitted_ops_dsa(131072, true);
        assert!(
            ops.contains(&(IndexScore as u16)),
            "full layer scores the indexer"
        );
        assert!(
            ops.contains(&(IndexSelect as u16)),
            "full layer selects top-k"
        );
        assert!(
            ops.contains(&(LayerNorm as u16)),
            "full layer k_norm LayerNorm"
        );
        assert!(ops.contains(&(FlashGatherDecode as u16)), "gather flash");
        assert!(
            !ops.contains(&(FlashMlaDecode as u16)),
            "no dense flash under DSA"
        );
    }

    #[test]
    fn glm_dsa_shared_layer_reuses_idx() {
        use DevOp::*;
        // ctx>CROSSOVER, 'shared': NO indexer ops (reuses the last full layer's idx) but still GATHERs.
        let ops = emitted_ops_dsa(131072, false);
        assert!(
            !ops.contains(&(IndexScore as u16)),
            "shared layer emits no score"
        );
        assert!(
            !ops.contains(&(IndexSelect as u16)),
            "shared layer emits no select"
        );
        assert!(
            !ops.contains(&(LayerNorm as u16)),
            "shared layer emits no k_norm"
        );
        assert!(
            ops.contains(&(FlashGatherDecode as u16)),
            "shared layer still gathers"
        );
    }

    #[test]
    fn glm_dense_block_sequence() {
        use DevOp::*;
        // Fused MLA (A+G): the 3 input GEMVs (q_a/kv_a/k_rope) -> one GemvQkv, and Wqa+Wqr -> one GemvQkv.
        let mla = vec![
            RmsNorm,
            GemvQkv,
            RmsNorm,
            GemvQkv,
            HeadNormRope,
            RmsNorm,
            HeadNormRope,
            FlashMlaDecode,
            MlaMergeFold,
            Gemv,
            Residual,
            RmsNorm,
        ];
        let mut want: Vec<u16> = mla.into_iter().map(|o| o as u16).collect();
        want.extend([DenseGluFp8Blk as u16, GemvFp8Blk as u16, Residual as u16]);
        assert_eq!(emitted_dense_ops(), want, "dense block op sequence");
        assert_eq!(emitted_dense_ops().len(), 15);
    }

    #[test]
    fn glm_block_op_count() {
        // 16 attention/pre-MoE ops after the A/G fusion (input q_a/kv_a/k_rope -> 1 GemvQkv, Wqa/Wqr
        // -> 1 GemvQkv; 2 dynamic-rope HeadNormRope + the 2-op router split + fused MlaMergeFold)
        // + 8*(glu+down) + 1 combine = 33 (was 36 pre-fusion).
        assert_eq!(emitted_ops(false).len(), 33);
    }

    // --- `--block` extraction path (M2, glm_build_block) ---------------------------
    // These exercise the actual single-block emit + descriptor build on the CPU with the
    // synthetic ref cfg (no checkpoint, no GPU): the block path must add NOTHING beyond
    // the validated per-layer block (no embed/tail), and the descriptor must reflect the
    // DSA IndexShare role + carried state.

    fn block_ops(c: &GlmCfg, ctx: u32, block: std::ops::Range<usize>) -> Vec<u16> {
        let (m, _desc) = glm_build_block(c, ctx, 256, block, true, "glm-ref", MlaArch::Glm);
        m.progs[0].insts.iter().map(|d| d.op).collect()
    }

    /// A single MoE-layer `--block 3` extraction emits EXACTLY the validated MoE block
    /// op sequence — no embed, no final-norm/lm_head/argmax tail. This is the numeric
    /// coverage lever: the block inherits glm_block_matches_reference_*'s GPU parity.
    #[test]
    fn glm_block_extract_matches_reference() {
        let c = glm_ref_cfg();
        assert_eq!(
            block_ops(&c, 512, 3..4),
            ref_sequence(true),
            "single-block --block 3 op sequence != validated MoE block"
        );
    }

    /// A multi-layer `--block 2..4` extraction is the per-layer blocks concatenated
    /// (dense layer 2 then MoE layer 3), and the residual ping-pong lands the output in
    /// `act.x` after an even layer count.
    #[test]
    fn glm_block_extract_multi_layer_chains() {
        let c = glm_ref_cfg();
        let mut want = emitted_dense_ops(); // layer 2 (dense)
        want.extend(ref_sequence(true)); // layer 3 (MoE)
        assert_eq!(block_ops(&c, 512, 2..4), want, "2-layer block != dense++moe");
        let (_, desc) = glm_build_block(&c, 512, 256, 2..4, true, "glm-ref", MlaArch::Glm);
        assert_eq!(desc.outputs[0].name, "act.x", "even layer count -> act.x out");
        assert_eq!(desc.layer, 2, "descriptor.layer = block start");
    }

    /// Descriptor for a single MoE block: arch/kind/dims + `act.xnext` output (odd
    /// layer count) + kv carried state, DSA gate OFF at this ctx (no dsa_indices).
    #[test]
    fn glm_block_descriptor_moe() {
        let c = glm_ref_cfg(); // indexer_full[3] = false (reuse)
        let (_, d) = glm_build_block(&c, 512, 256, 3..4, true, "glm-ref", MlaArch::Glm);
        assert_eq!(d.arch, "glm_mla_dsa");
        assert_eq!(d.kind, vec!["mla_dsa", "moe_ffn"]);
        assert_eq!(d.dtype, "fp8");
        assert_eq!(d.dims.kv_lora, Some(512));
        assert_eq!(d.dims.q_lora, Some(2048));
        assert_eq!(d.dims.n_exp, Some(256));
        assert_eq!(d.dims.top_k, Some(8));
        assert_eq!(d.dims.shared_exp, Some(1));
        assert_eq!(d.dims.moe_inter, Some(2048));
        assert_eq!(d.dims.index_topk, Some(2048));
        assert_eq!(d.outputs[0].name, "act.xnext", "odd layer count -> act.xnext");
        assert_eq!(d.weights.prefix, "model.layers.3.");
        assert!(d.programs.prefill_buckets.is_empty(), "GLM is decode-only");
        // DSA gate off (ctx <= CROSSOVER): reuse role, but NO dsa_indices carried.
        assert_eq!(d.dsa_role.as_deref(), Some("reuse"));
        assert_eq!(d.carried_state.len(), 1);
        assert_eq!(d.carried_state[0].role, "kv");
        assert_eq!(d.carried_state[0].tensors, vec!["kv.3.ckv", "kv.3.krot"]);
    }

    /// Descriptor for a DENSE block (`--block 0`): no MoE dims, dense_ffn kind.
    #[test]
    fn glm_block_descriptor_dense() {
        let c = glm_ref_cfg();
        let (_, d) = glm_build_block(&c, 512, 256, 0..1, true, "glm-ref", MlaArch::Glm);
        assert_eq!(d.kind, vec!["mla_dsa", "dense_ffn"]);
        assert_eq!(d.dims.n_exp, None, "dense block has no MoE dims");
        assert_eq!(d.dims.moe_inter, None);
        assert_eq!(d.dims.kv_lora, Some(512), "MLA dims still present");
    }

    /// IndexShare (§7): under an ARMED DSA gate (ctx > CROSSOVER=65536), a 'reuse'
    /// layer carries `dsa_indices` in (it does not recompute the top-k), while an
    /// 'indexer' layer computes them in-block (kv carries its kidx cache instead).
    #[test]
    fn glm_block_dsa_indexshare_carried_state() {
        // 'reuse' layer 3 (indexer_types[3] = shared).
        let mut c = glm_ref_cfg();
        c.indexer_full = vec![false, false, false, false];
        let (_, reuse) = glm_build_block(&c, 131072, 256, 3..4, true, "glm-ref", MlaArch::Glm);
        assert_eq!(reuse.dsa_role.as_deref(), Some("reuse"));
        let dsa = reuse
            .carried_state
            .iter()
            .find(|s| s.role == "dsa_indices")
            .expect("reuse layer carries dsa_indices");
        assert_eq!(dsa.tensors, vec!["act.iidx"]);

        // 'indexer' layer 3 (indexer_types[3] = full): computes indices in-block, so
        // no dsa_indices carry; its kidx key cache joins the kv carried state.
        c.indexer_full = vec![false, false, false, true];
        let (_, idx) = glm_build_block(&c, 131072, 256, 3..4, true, "glm-ref", MlaArch::Glm);
        assert_eq!(idx.dsa_role.as_deref(), Some("indexer"));
        assert!(
            idx.carried_state.iter().all(|s| s.role != "dsa_indices"),
            "indexer layer does not carry dsa_indices in"
        );
        assert!(
            idx.carried_state[0].tensors.contains(&"kv.3.kidx".to_string()),
            "indexer layer carries its kidx cache"
        );
    }

    /// The MLA flash-decode split factor is the ctx-scaled cost optimum, capped by the ACTUAL
    /// per-rank chip-fill `fill = ceil(n_cu / (nh_l/GF))` and the KV-tile count. `glm_nsplit` takes
    /// nh_l (= n_head/tp) so the cap is correct under TP/EP — the pre-fix bug sized it from the
    /// global n_head=64, pinning the cap to tp=1's fill regardless of TP. Asserts the caps and the
    /// measured (MI350X mla_perf) chain optima: ns~16 up to 8k, ns~64 at 32k.
    #[test]
    fn glm_nsplit_is_ctx_scaled_and_capped_per_rank() {
        let n_cu = 256u32;
        for &(_tp, nh_l) in &[(1u32, 64u32), (2, 32), (4, 16), (8, 8)] {
            let n_grp = (nh_l / GLM_MLA_GF).max(1);
            let fill = (n_cu + n_grp - 1) / n_grp;
            let mut prev = 0u32;
            for &ctx in &[1024u32, 4096, 8192, 16384, 32768, 65536, 131072] {
                let ns = glm_nsplit(ctx, nh_l);
                let kv_tiles = ctx.div_ceil(32);
                // Cap 1 — never over-split past the chip (the nh_l-aware fill).
                assert!(
                    ns <= fill,
                    "nh_l={nh_l} ctx={ctx}: ns={ns} exceeds chip-fill {fill}"
                );
                // Cap 2 — never split finer than there are KV tiles (no empty splits).
                assert!(
                    ns <= kv_tiles,
                    "nh_l={nh_l} ctx={ctx}: ns={ns} exceeds {kv_tiles} KV tiles"
                );
                // Monotone non-decreasing in ctx (more latent => more useful splits).
                assert!(
                    ns >= prev,
                    "nh_l={nh_l} ctx={ctx}: ns={ns} < prev {prev} (not ctx-monotone)"
                );
                prev = ns;
            }
        }
        // Measured chain optima locked in (fill-permitting): ns=16 up to 8k, ns=64 at 32k.
        for &nh_l in &[8u32, 16] {
            assert_eq!(
                glm_nsplit(1024, nh_l),
                16,
                "nh_l={nh_l}: 1k optimum is ns=16"
            );
            assert_eq!(
                glm_nsplit(8192, nh_l),
                16,
                "nh_l={nh_l}: 8k optimum is ns=16"
            );
            assert_eq!(
                glm_nsplit(32768, nh_l),
                64,
                "nh_l={nh_l}: 32k optimum is ns=64"
            );
        }
        // tp=1 is chip-full at ns=16 (n_grp=16), so the fill cap pins every ctx to 16 — byte-identical
        // to the pre-fix path (no regression on single-GPU decode).
        for &ctx in &[1024u32, 8192, 32768, 131072] {
            assert_eq!(
                glm_nsplit(ctx, 64),
                16,
                "tp=1 ctx={ctx}: fill-capped to 16 (unchanged)"
            );
        }
        // The refined rule must NOT full-fill mid ctx (the measured 8k regression at ns=128): at tp=8
        // 8k it stays at the floor, not fill=128.
        assert!(
            glm_nsplit(8192, 8) < ((256 + 1) / 2),
            "tp=8 8k must not full-fill (mid-ctx merge regression)"
        );
    }

    #[test]
    fn glm_cfg_qk_scale() {
        let c = glm_ref_cfg();
        assert_eq!(c.qk_head(), 256);
        assert!(
            (c.attn_scale - 0.0625).abs() < 1e-6,
            "MLA scale = 1/sqrt(256)"
        );
        assert!(
            c.is_dense(0) && c.is_dense(2) && !c.is_dense(3),
            "first_k_dense_replace=3"
        );
    }
}

#[cfg(test)]
mod kimi_tests {
    //! Kimi K2.7 / DeepSeek MLA+MoE `--block` extraction (M3, plans/block-asset-harness.md §5.0/
    //! §5.3/§7). Kimi REUSES the GLM MLA + MoE emit verbatim (glm_build_block) with a cfg that holds
    //! the DSA gate off (`has_dsa=false`) — so a Kimi block is the SAME op sequence as a GLM block
    //! BELOW the DSA crossover, minus every indexer artifact: no DSA scratch, FlashMlaDecode (never
    //! FlashGatherDecode) at ANY ctx, and a descriptor with no dsa_role / no index_* dims. These
    //! synthetic-CPU tests are the only verification available on this box (no Kimi checkpoint, no
    //! transformers → no real blob, no GPU parity). They lock in the op sequence + descriptor exactly
    //! as glm_tests does for GLM.
    use super::*;

    /// Synthetic small Kimi cfg (structurally faithful: DeepSeek-schema MLA + MoE, first_k_dense=1
    /// so layer 0 is dense and 1+ are MoE, has_dsa=false). Real K2.7 geometry is hidden 7168 / 64
    /// heads / kv_lora 512 / q_lora 1536 / qk_nope 128 / qk_rope 64 / v_head 128 / 384 exp / top_k 8
    /// / moe_inter 2048; the shape logic is dim-agnostic, so small dims exercise the same emit.
    fn kimi_ref_cfg() -> GlmCfg {
        GlmCfg {
            layers: 4,
            hidden: 256,
            heads: 4,
            kv_lora: 64,
            q_lora: 96,
            qk_nope: 32,
            qk_rope: 16,
            v_head: 32,
            vocab: 1000,
            eps: 1e-5,
            n_exp: 16,
            top_k: 4,
            moe_inter: 128,
            dense_inter: 256,
            first_k_dense: 1,
            route_scale: 2.5,
            attn_scale: (48f32).powf(-0.5), // 1/sqrt(qk_nope+qk_rope = 48)
            rope_theta: 50_000.0,
            tp: 1,
            ep: false,
            group: false,
            // Indexer fields are inert under has_dsa=false (never read); set placeholders.
            index_heads: 8,
            index_dim: 32,
            index_topk: 64,
            indexer_full: Vec::new(), // Kimi/DeepSeek config has no `indexer_types`
            has_dsa: false,
        }
    }

    fn block_ops(c: &GlmCfg, ctx: u32, block: std::ops::Range<usize>, arch: MlaArch) -> Vec<u16> {
        let (m, _d) = glm_build_block(c, ctx, 256, block, true, "kimi-ref", arch);
        m.progs[0].insts.iter().map(|d| d.op).collect()
    }

    /// Expected MoE-block op sequence: shared MLA (12 ops) + router split (2) + shared expert (2) +
    /// top_k×(glu, down) + MoeCombine. IDENTICAL shape to glm_tests::ref_sequence but parameterized
    /// on top_k — the reuse the arch is built on.
    fn kimi_moe_sequence(use_fp8: bool, top_k: usize) -> Vec<u16> {
        use DevOp::*;
        let (glu, down) = if use_fp8 {
            (MoeExpertGluFp8Blk, MoeExpertDownFp8Blk)
        } else {
            (MoeExpertGlu, MoeExpertDown)
        };
        let mut ops = vec![
            RmsNorm,        // input_layernorm
            GemvQkv,        // FUSED A: q_a + kv_a + k_rope down-projections
            RmsNorm,        // q_a_layernorm
            GemvQkv,        // FUSED G: q_absorb + q_rope down
            HeadNormRope,   // q_rope dynamic interleaved RoPE
            RmsNorm,        // kv_a_layernorm -> latent cache
            HeadNormRope,   // k_rope dynamic RoPE -> rope cache
            FlashMlaDecode, // MLA flash (NO DSA gather)
            MlaMergeFold,   // fused latent merge + W_uv fold
            Gemv,           // o_proj
            Residual,       // post-attn residual
            RmsNorm,        // post_attention_layernorm
            Gemv,           // router score GEMV
            MoeRouterTopk,  // router top-k select
            GemvGlu,        // shared expert gate|up
            Gemv,           // shared expert down
        ];
        for _ in 0..top_k {
            ops.push(glu);
            ops.push(down);
        }
        ops.push(MoeCombine);
        ops.into_iter().map(|o| o as u16).collect()
    }

    /// Expected DENSE-block op sequence: shared MLA (12) + block-fp8 SwiGLU (gate/up + down) +
    /// residual. The GLM emitter's dense FFN is block-fp8 regardless of `use_fp8`, so Kimi's dense
    /// layer (layer 0) inherits those opcodes.
    fn kimi_dense_sequence() -> Vec<u16> {
        use DevOp::*;
        vec![
            RmsNorm,
            GemvQkv,
            RmsNorm,
            GemvQkv,
            HeadNormRope,
            RmsNorm,
            HeadNormRope,
            FlashMlaDecode,
            MlaMergeFold,
            Gemv,
            Residual,
            RmsNorm,
            DenseGluFp8Blk,
            GemvFp8Blk,
            Residual,
        ]
        .into_iter()
        .map(|o| o as u16)
        .collect()
    }

    /// A single MoE-layer `--block 1` extraction emits EXACTLY the MLA+MoE block — no embed, no
    /// final-norm/lm_head/argmax tail, `act.x` in and out.
    #[test]
    fn kimi_block_extract_matches_mla_moe_sequence() {
        let c = kimi_ref_cfg();
        assert_eq!(
            block_ops(&c, 512, 1..2, MlaArch::Kimi),
            kimi_moe_sequence(true, 4),
            "single-block --block 1 op sequence != MLA+MoE block (fp8)"
        );
        assert_eq!(
            {
                let (m, _) = glm_build_block(&c, 512, 256, 1..2, false, "kimi-ref", MlaArch::Kimi);
                m.progs[0].insts.iter().map(|d| d.op).collect::<Vec<_>>()
            },
            kimi_moe_sequence(false, 4),
            "bf16 op sequence != MLA+MoE block"
        );
    }

    /// Descriptor for a Kimi MoE block: arch tag, mla_attn+moe_ffn kind, NO dsa_role, MLA+MoE dims,
    /// NO index_* dims, KV latent (ckv/krot) carried state only, decode-only programs.
    #[test]
    fn kimi_block_descriptor_moe() {
        let c = kimi_ref_cfg();
        let (_, d) = glm_build_block(&c, 512, 256, 1..2, true, "kimi-k2.7", MlaArch::Kimi);
        assert_eq!(d.arch, "kimi_mla_moe");
        assert_eq!(d.kind, vec!["mla_attn", "moe_ffn"]);
        assert_eq!(d.dtype, "fp8");
        assert_eq!(d.dsa_role, None, "plain MLA has no DSA indexer role");
        assert_eq!(d.dims.heads, Some(4));
        assert_eq!(d.dims.kv_lora, Some(64));
        assert_eq!(d.dims.q_lora, Some(96));
        assert_eq!(d.dims.n_exp, Some(16));
        assert_eq!(d.dims.top_k, Some(4));
        assert_eq!(d.dims.shared_exp, Some(1));
        assert_eq!(d.dims.moe_inter, Some(128));
        assert_eq!(d.dims.index_heads, None, "no DSA => no index dims");
        assert_eq!(d.dims.index_dim, None);
        assert_eq!(d.dims.index_topk, None);
        assert_eq!(d.layer, 1);
        assert_eq!(d.weights.prefix, "model.layers.1.");
        assert_eq!(d.outputs[0].name, "act.xnext", "odd layer count -> act.xnext");
        assert!(
            d.programs.prefill_buckets.is_empty(),
            "GLM/Kimi emit is decode-only"
        );
        assert_eq!(d.programs.decode_t, 1);
        // KV latent carried state only — no kidx, no dsa_indices.
        assert_eq!(d.carried_state.len(), 1);
        assert_eq!(d.carried_state[0].role, "kv");
        assert_eq!(d.carried_state[0].layout, "mla_latent");
        assert_eq!(
            d.carried_state[0].tensors,
            vec!["kv.1.ckv", "kv.1.krot"],
            "MLA latent caches only (no indexer kidx)"
        );
    }

    /// Descriptor for a Kimi DENSE block (layer 0, first_k_dense=1): dense_ffn kind, no MoE dims,
    /// MLA dims still present.
    #[test]
    fn kimi_block_descriptor_dense() {
        let c = kimi_ref_cfg();
        let (_, d) = glm_build_block(&c, 512, 256, 0..1, true, "kimi-ref", MlaArch::Kimi);
        assert_eq!(d.kind, vec!["mla_attn", "dense_ffn"]);
        assert_eq!(d.dims.n_exp, None, "dense block has no MoE dims");
        assert_eq!(d.dims.moe_inter, None);
        assert_eq!(d.dims.kv_lora, Some(64), "MLA dims still present");
        assert_eq!(d.dsa_role, None);
    }

    /// A multi-layer `--block 0..2` extraction chains dense layer 0 then MoE layer 1, and the
    /// residual ping-pong lands the output back in `act.x` after an even layer count.
    #[test]
    fn kimi_block_multi_layer_chains() {
        let c = kimi_ref_cfg();
        let mut want = kimi_dense_sequence(); // layer 0 (dense)
        want.extend(kimi_moe_sequence(true, 4)); // layer 1 (MoE)
        assert_eq!(
            block_ops(&c, 512, 0..2, MlaArch::Kimi),
            want,
            "2-layer block != dense++moe"
        );
        let (_, d) = glm_build_block(&c, 512, 256, 0..2, true, "kimi-ref", MlaArch::Kimi);
        assert_eq!(d.outputs[0].name, "act.x", "even layer count -> act.x out");
        assert_eq!(d.layer, 0, "descriptor.layer = block start");
    }

    /// The DSA gate is held OFF at EVERY ctx (has_dsa=false): even at 131072 (well past GLM's 65536
    /// crossover) the block emits FlashMlaDecode — never FlashGatherDecode — and carries no
    /// dsa_indices / no kidx. This is what "reuse GLM MLA without DSA" means structurally.
    #[test]
    fn kimi_no_dsa_at_long_ctx() {
        let c = kimi_ref_cfg();
        let ops = block_ops(&c, 131072, 1..2, MlaArch::Kimi);
        assert!(
            ops.contains(&(DevOp::FlashMlaDecode as u16)),
            "dense MLA flash present"
        );
        assert!(
            !ops.contains(&(DevOp::FlashGatherDecode as u16)),
            "no DSA gather flash for Kimi"
        );
        let (_, d) = glm_build_block(&c, 131072, 256, 1..2, true, "kimi-ref", MlaArch::Kimi);
        assert_eq!(d.dsa_role, None);
        assert!(
            d.carried_state.iter().all(|s| s.role != "dsa_indices"),
            "no dsa_indices carried"
        );
        assert!(
            d.carried_state[0].tensors.iter().all(|t| !t.contains("kidx")),
            "no indexer kidx cache"
        );
    }

    /// The DeepSeek flavor differs only in the descriptor arch tag; the emit + kind + no-DSA are
    /// identical to Kimi.
    #[test]
    fn deepseek_arch_tag() {
        let c = kimi_ref_cfg();
        let (_, d) = glm_build_block(&c, 512, 256, 1..2, true, "deepseek-v3", MlaArch::DeepSeek);
        assert_eq!(d.arch, "deepseek_mla_moe");
        assert_eq!(d.kind, vec!["mla_attn", "moe_ffn"]);
        assert_eq!(d.dsa_role, None);
    }
}

#[cfg(test)]
mod gemma_router_emit_tests {
    use super::*;

    fn router_program(split_plan: Option<(u32, DevOp)>) -> packet::devbuild::Program {
        router_program_b(split_plan, 1)
    }

    fn router_program_b(
        split_plan: Option<(u32, DevOp)>,
        nrow: u32,
    ) -> packet::devbuild::Program {
        let mut b = Builder::new(188);
        let dep = b.emit(DevOp::Nop, vec![0], &[], |_| {});
        let _ = emit_gemma_moe_router(
            &mut b,
            dep,
            10, // table
            11, // residual
            12, // projection
            13, // channel scale
            14, // per-expert scale
            if split_plan.is_some() {
                15
            } else {
                TENSOR_NONE
            },
            2816,
            128,
            8,
            (2816.0f32).powf(-0.5),
            1e-6,
            split_plan,
            nrow,
        );
        b.finish()
    }

    /// BATCH B>1: the batch row count reaches every router op, top-k gets one CTA per row, and
    /// the score op's CTA count scales with the (row, expert) pair space.
    #[test]
    fn batched_router_emit_carries_b() {
        let plan = gemma_moe_router_split_plan(188, 128, 8);
        let p = router_program_b(plan, 8);
        let score = &p.insts[1];
        let topk = &p.insts[2];
        assert_eq!(score.i[2], 8, "score op carries B");
        assert_eq!(score.blocks, 128, "8 rows x 128 experts / 8 per CTA");
        assert_eq!(topk.i[3], 8, "top-k carries B");
        assert_eq!(topk.blocks, 8, "one top-k CTA per row");
        // B=1 must leave the immediate at 0 so the packet bytes never move.
        let p1 = router_program(gemma_moe_router_split_plan(188, 128, 1));
        assert_eq!(p1.insts[1].i[2], 0);
        assert_eq!(p1.insts[2].i[3], 0);
        assert_eq!(p1.insts[2].blocks, 1);
    }

    #[test]
    fn legacy_router_emit_stays_one_original_opcode() {
        let p = router_program(None);
        assert_eq!(p.insts.len(), 2);
        let r = &p.insts[1];
        assert_eq!(r.op, DevOp::MoeRouterGemma as u16);
        assert_eq!(r.blocks, 1);
        assert_eq!(r.t[..5], [10, 11, 12, 13, 14]);
        assert_eq!(r.i[..3], [2816, 128, 8]);
    }

    #[test]
    fn split_router_emits_parallel_score_then_one_cta_tail() {
        let p = router_program(Some((16, DevOp::MoeRouterGemmaScore)));
        assert_eq!(p.insts.len(), 3);
        let score = &p.insts[1];
        let tail = &p.insts[2];
        assert_eq!(score.op, DevOp::MoeRouterGemmaScore as u16);
        assert_eq!(score.blocks, 16);
        assert_eq!(score.t[..4], [15, 11, 12, 13]);
        assert_eq!(tail.op, DevOp::MoeRouterGemmaTopk as u16);
        assert_eq!(tail.blocks, 1);
        assert_eq!(tail.t[..3], [10, 15, 14]);
        assert_eq!(tail.wait_len, 1);
        assert_eq!(p.waits[tail.wait_ofs as usize].threshold, 16);
    }

    #[test]
    fn fast_router_is_a_distinct_default_off_score_opcode() {
        let p = router_program(Some((16, DevOp::MoeRouterGemmaScoreFast)));
        assert_eq!(p.insts[1].op, DevOp::MoeRouterGemmaScoreFast as u16);
        assert_eq!(p.insts[1].blocks, 16);
        assert_eq!(p.insts[2].op, DevOp::MoeRouterGemmaTopk as u16);
    }
}

#[cfg(test)]
mod mode_tests {
    //! `Mode` exists to split the old `decode: bool` into two independent axes. The ONLY thing
    //! that keeps that refactor honest is that the two pre-existing corners still decode to the
    //! same pair of booleans they were hardcoded to before — `Prefill` was `decode=false`
    //! everywhere, `Decode` was `decode=true` everywhere. If either row below changes, every
    //! emitted program changes with it, silently. (Verified once against real packets: the
    //! Qwen3-4B blob is byte-identical pre/post refactor at ctx 4256/16544 and n_cu 170/256.)
    use super::Mode;

    #[test]
    fn legacy_corners_are_unchanged() {
        assert!(!Mode::Prefill.decode_shape() && !Mode::Prefill.gemv());
        assert!(Mode::Decode.decode_shape() && Mode::Decode.gemv());
    }

    #[test]
    fn decode_tiled_is_decode_shape_on_prefill_kernels() {
        // The whole point: decode's shape (one row, KV append, ring mask) with prefill's
        // kernels (tiled GEMM, FlashPrefill). Neither legacy corner can express this.
        assert!(Mode::DecodeTiled.decode_shape());
        assert!(!Mode::DecodeTiled.gemv());
    }
}

#[cfg(test)]
mod pick_tile_tests {
    //! The hwspec-driven picker is a STATIC, shape-agnostic choice — so it is testable
    //! offline, with no GPU. These lock in the tile chosen for every projection of the three
    //! supported architectures at the prefill chunk sizes that matter, proving the picker both
    //! fills the CUs on the underutilized shapes AND does not regress the ones that already
    //! saturate. `n_cu = 256` (MI350X).
    use super::{gemm_lds_bytes, hwspec, pick_tile, DevOp, GFX950_TILES};
    use costmodel::cost::{dma_cycles, macs_cycles};
    use costmodel::MmaDtype;

    const N_CU: u32 = 256;
    fn pt(m: u32, n: u32, k: u32) -> DevOp {
        pick_tile(m, n, k, N_CU)
    }

    /// The picker exactly as it was before selection moved behind the capability
    /// registry: one loop over a constant table, first-match-wins on ties.
    ///
    /// Kept as the differential reference. The assertions below pin the shapes
    /// that were reasoned about by hand; this pins everything else, which is
    /// what actually rules out a silent regression on some shape nobody listed.
    fn pick_tile_legacy(m: u32, n: u32, k: u32, n_cu: u32) -> DevOp {
        let spec = hwspec::registry::lookup("MI350X").expect("gfx950 spec in registry");
        let lds_budget = spec.sm.shared_mem.0;
        let (m, n, k) = (m as u64, n as u64, k as u64);
        let n_cu = (n_cu as u64).max(1);

        let mut best = (DevOp::Gemm, u64::MAX);
        for (op, bm, bn, bk) in GFX950_TILES {
            if gemm_lds_bytes(bm, bn, bk) > lds_budget {
                continue;
            }
            let tiles = m.div_ceil(bm) * n.div_ceil(bn);
            let rounds = tiles.div_ceil(n_cu);
            let k_iters = k.div_ceil(bk);
            let compute = k_iters * macs_cycles(spec, bm * bn * bk, MmaDtype::Bf16);
            let dma = dma_cycles(spec, (bm * k + k * bn) * 2, false);
            let cost = rounds.saturating_mul(compute.max(dma));
            if cost < best.1 {
                best = (op, cost);
            }
        }
        best.0
    }

    /// Routing selection through the registry must not change a single answer on
    /// the hardware the old picker was written for. Swept rather than sampled:
    /// tie-breaking was the real risk, since the old loop preferred the larger
    /// tile by table order while opcode order would put `GemmSmall` (14) ahead of
    /// `GemmMed` (15).
    #[test]
    fn registry_selection_matches_the_legacy_picker_everywhere() {
        let ms = [1u32, 8, 16, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384];
        let ns = [128u32, 512, 1024, 2048, 2560, 4096, 5376, 8192, 9728, 14336, 16384, 21504];
        let ks = [128u32, 512, 2560, 4096, 5376, 8192, 14336, 21504];
        let cus = [1u32, 64, 128, 256, 304];

        let mut checked = 0usize;
        for &m in &ms {
            for &n in &ns {
                for &k in &ks {
                    for &n_cu in &cus {
                        let want = pick_tile_legacy(m, n, k, n_cu);
                        let got = pick_tile(m, n, k, n_cu);
                        assert_eq!(
                            got, want,
                            "diverged at m={m} n={n} k={k} n_cu={n_cu}: \
                             registry chose {got:?}, legacy chose {want:?}"
                        );
                        checked += 1;
                    }
                }
            }
        }
        assert_eq!(checked, ms.len() * ns.len() * ks.len() * cus.len());
    }

    #[test]
    fn llama31_8b_prefill_4k() {
        // hidden 4096, inter 14336, heads 32, kv_heads 8, hd 128.
        // q/o saturate 256 CUs at 256x256 (16x16 = 256 tiles) — keep the big tile.
        assert_eq!(pt(4096, 4096, 4096), DevOp::Gemm, "q_proj");
        assert_eq!(pt(4096, 4096, 4096), DevOp::Gemm, "o_proj");
        // k/v (N=1024) are only 16x4 = 64 tiles at 256x256 — a QUARTER of the machine. The
        // picker drops to 128x128 (16x8 = 256 tiles) to fill all 256 CUs. This is the fix the
        // old heuristic missed (it pinned k/v to 256x256, blind to CU fill).
        assert_eq!(pt(4096, 1024, 4096), DevOp::GemmMed, "k_proj / v_proj");
        // gate/up (fused GemmGlu path keys off Gemm) and down saturate — keep 256x256.
        assert_eq!(pt(4096, 14336, 4096), DevOp::Gemm, "gate/up (fused)");
        assert_eq!(pt(4096, 4096, 14336), DevOp::Gemm, "down_proj");
    }

    #[test]
    fn llama31_8b_prefill_8k_kv_already_half_full() {
        // At M=8192 k/v already make 32x4 = 128 tiles (half fill) at 256x256; splitting to
        // 128x128 would need 2 rounds for equal cost, so the higher-intensity 256x256 stays.
        assert_eq!(pt(8192, 1024, 4096), DevOp::Gemm, "k/v at 8k");
    }

    #[test]
    fn qwen3_4b_prefill_4k() {
        // hidden 2560, inter 9728, heads 32, kv_heads 8, hd 128.
        assert_eq!(pt(4096, 4096, 2560), DevOp::Gemm, "q_proj");
        assert_eq!(
            pt(4096, 1024, 2560),
            DevOp::GemmMed,
            "k_proj / v_proj (fill)"
        );
        assert_eq!(pt(4096, 9728, 2560), DevOp::Gemm, "gate/up");
        assert_eq!(pt(4096, 2560, 9728), DevOp::Gemm, "down_proj");
    }

    #[test]
    fn gemma31b_no_regression() {
        // hidden 5376, inter 21504. Gemma's kv projections are WIDE (sliding N=4096, global
        // N=2048), so they already saturate — the picker must keep 256x256 everywhere it did
        // before. No Gemma projection is small enough to reselect.
        assert_eq!(pt(4096, 8192, 5376), DevOp::Gemm, "q sliding");
        assert_eq!(pt(4096, 16384, 5376), DevOp::Gemm, "q global");
        assert_eq!(pt(4096, 4096, 5376), DevOp::Gemm, "kv sliding (N=4096)");
        assert_eq!(pt(4096, 2048, 5376), DevOp::Gemm, "kv global (N=2048)");
        assert_eq!(pt(4096, 5376, 8192), DevOp::Gemm, "o sliding");
        assert_eq!(pt(4096, 21504, 5376), DevOp::Gemm, "gate/up");
        assert_eq!(pt(4096, 5376, 21504), DevOp::Gemm, "down");
    }

    #[test]
    fn short_prompt_buckets_use_narrow_tiles() {
        // A 128-row chunk cannot fill 256 CUs with a 256x256 tile (q_proj = 1x16 = 16 tiles),
        // so the picker drops to the narrow-M kernels — matching the measured T=128 optima in
        // op_gemm.h (64x128 fastest for the tall projections at small M).
        assert_eq!(pt(128, 8192, 5376), DevOp::GemmSmall, "T=128 q sliding");
        assert_ne!(
            pt(128, 4096, 4096),
            DevOp::Gemm,
            "T=128 must not pick the big tile"
        );
    }
}

/// Nemotron-3 Mamba-2 hybrid tests (plans/block-asset-harness.md §7 Nemotron, §11 M4).
///
/// Two kinds of check, matching what CAN be verified on this box (no Nemotron checkpoint, no
/// transformers, no GPU):
///  1. **Numeric golden for the NEW SSM math.** The selective SSD scan is implemented two ways —
///     the STATEFUL recurrence the device kernel (op_mamba.cuh) and the block emit mirror, and an
///     INDEPENDENT closed-form dual (materialize the per-(t,s) decay and sum) — and asserted equal
///     to f32 tolerance. This is the real correctness lever: no torch dependency, a direct
///     sequential/quadratic pair is the golden.
///  2. **Emit op-sequence + descriptor** (synthetic CPU, like glm_tests / kimi_tests): the `--block`
///     extraction emits exactly the expected DevOp sequence per layer kind (mamba / gqa_attn / moe)
///     and the arch-agnostic descriptor (kind, dims, carried_state) is correct.
#[cfg(test)]
mod nemotron_tests {
    use super::*;

    // ---- reference Mamba-2 SSD math (f32) ------------------------------------------------------

    fn silu(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }
    fn softplus(x: f32) -> f32 {
        // numerically-stable log(1+e^x)
        if x > 20.0 {
            x
        } else {
            (1.0 + x.exp()).ln()
        }
    }

    /// Deterministic pseudo-random stream in [-amp, amp] (reproducible, no rand dep).
    struct Lcg(u64);
    impl Lcg {
        fn f(&mut self, amp: f32) -> f32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = ((self.0 >> 33) as f32) / ((1u64 << 31) as f32); // [0,1)
            (u * 2.0 - 1.0) * amp
        }
    }

    struct Dims {
        t: usize,
        n_head: usize,
        head_dim: usize,
        d_state: usize,
        n_groups: usize,
    }
    impl Dims {
        fn d_inner(&self) -> usize {
            self.n_head * self.head_dim
        }
        fn hpg(&self) -> usize {
            self.n_head / self.n_groups
        }
    }

    /// SSD selective scan, STATEFUL recurrence form (what the device kernel / block emit mirror).
    /// `x` [T, d_inner], `b`/`cc` [T, n_groups*d_state], `dt_eff` [T, n_head] (already softplus'd),
    /// `a` [n_head] (= -exp(A_log)), `dd` [n_head] (the D skip). `ssm` [n_head*head_dim*d_state] is
    /// read as the initial state and OVERWRITTEN with the final state. Returns yscan [T, d_inner].
    fn scan_recurrence(
        d: &Dims,
        x: &[f32],
        b: &[f32],
        cc: &[f32],
        dt_eff: &[f32],
        a: &[f32],
        dd: &[f32],
        ssm: &mut [f32],
    ) -> Vec<f32> {
        let (nh, hd, ds, ng) = (d.n_head, d.head_dim, d.d_state, d.n_groups);
        let di = d.d_inner();
        let hpg = d.hpg();
        let mut y = vec![0.0f32; d.t * di];
        for t in 0..d.t {
            for h in 0..nh {
                let dtv = dt_eff[t * nh + h];
                let da = (dtv * a[h]).exp();
                let g = h / hpg;
                for p in 0..hd {
                    let xv = x[t * di + h * hd + p];
                    let mut acc = 0.0f32;
                    for n in 0..ds {
                        let bn = b[t * ng * ds + g * ds + n];
                        let cn = cc[t * ng * ds + g * ds + n];
                        let si = h * hd * ds + p * ds + n;
                        ssm[si] = da * ssm[si] + dtv * xv * bn;
                        acc += cn * ssm[si];
                    }
                    y[t * di + h * hd + p] = acc + dd[h] * xv;
                }
            }
        }
        y
    }

    /// SSD selective scan, INDEPENDENT closed-form dual: h_t = exp(cum_t)·h_init +
    /// Σ_{s≤t} exp(cum_t − cum_s)·dt_s·x_s⊗B_s, y_t = Σ_n C_t·h_t + D·x_t. Materializes the decay
    /// per (t,s) and sums — a structurally different computation (different float order) than the
    /// stateful recurrence, so agreement to tolerance validates the recurrence. `ssm_init` is the
    /// carried-in state; does NOT mutate it.
    fn scan_dual(
        d: &Dims,
        x: &[f32],
        b: &[f32],
        cc: &[f32],
        dt_eff: &[f32],
        a: &[f32],
        dd: &[f32],
        ssm_init: &[f32],
    ) -> Vec<f32> {
        let (nh, hd, ds, ng) = (d.n_head, d.head_dim, d.d_state, d.n_groups);
        let di = d.d_inner();
        let hpg = d.hpg();
        let mut y = vec![0.0f32; d.t * di];
        for h in 0..nh {
            // cumulative log-decay per t: cum[t] = Σ_{r=0}^{t} dt_r·A_h
            let mut cum = vec![0.0f32; d.t];
            let mut run = 0.0f32;
            for t in 0..d.t {
                run += dt_eff[t * nh + h] * a[h];
                cum[t] = run;
            }
            let g = h / hpg;
            for t in 0..d.t {
                for p in 0..hd {
                    let mut acc = dd[h] * x[t * di + h * hd + p];
                    for n in 0..ds {
                        let cn = cc[t * ng * ds + g * ds + n];
                        // initial-state contribution
                        let mut hval = cum[t].exp() * ssm_init[h * hd * ds + p * ds + n];
                        // input contributions from all s ≤ t
                        for s in 0..=t {
                            let decay = (cum[t] - cum[s]).exp();
                            let xs = x[s * di + h * hd + p];
                            let bs = b[s * ng * ds + g * ds + n];
                            hval += decay * dt_eff[s * nh + h] * xs * bs;
                        }
                        acc += cn * hval;
                    }
                    y[t * di + h * hd + p] = acc;
                }
            }
        }
        y
    }

    fn max_abs(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
    }

    /// The NEW SSM math: the stateful recurrence (kernel/emit form) equals the independent
    /// closed-form dual to f32 tolerance — with a NON-ZERO carried-in ssm_state, so the initial
    /// state term is exercised. Reports the max-abs error vs the golden.
    #[test]
    fn mamba2_scan_matches_independent_recurrence() {
        let d = Dims { t: 6, n_head: 4, head_dim: 5, d_state: 3, n_groups: 2 };
        let di = d.d_inner();
        let gd = d.n_groups * d.d_state;
        let mut r = Lcg(0x1234_5678_9abc_def0);
        let x: Vec<f32> = (0..d.t * di).map(|_| r.f(0.5)).collect();
        let b: Vec<f32> = (0..d.t * gd).map(|_| r.f(0.5)).collect();
        let cc: Vec<f32> = (0..d.t * gd).map(|_| r.f(0.5)).collect();
        // dt already softplus'd (positive); A = -exp(a_log) (negative) => stable decay in (0,1).
        let dt_eff: Vec<f32> = (0..d.t * d.n_head).map(|_| softplus(r.f(1.0))).collect();
        let a: Vec<f32> = (0..d.n_head).map(|_| -(r.f(0.5) + 0.7).exp()).collect();
        let dd: Vec<f32> = (0..d.n_head).map(|_| r.f(0.5)).collect();
        let ssm_init: Vec<f32> = (0..d.n_head * d.head_dim * d.d_state).map(|_| r.f(0.3)).collect();

        let mut ssm = ssm_init.clone();
        let y_rec = scan_recurrence(&d, &x, &b, &cc, &dt_eff, &a, &dd, &mut ssm);
        let y_dual = scan_dual(&d, &x, &b, &cc, &dt_eff, &a, &dd, &ssm_init);
        let err = max_abs(&y_rec, &y_dual);
        eprintln!("mamba2 SSM scan: max-abs err (recurrence vs independent dual) = {err:e}");
        assert!(err < 1e-4, "SSM scan diverges from independent golden: max-abs {err:e}");
    }

    /// Prefill/decode equivalence: running the scan as ONE T-step prefill leaves the same
    /// ssm_state, and yields the same last-token output, as feeding the tokens one at a time
    /// through single-step decode calls (each carrying the state forward). This is the
    /// state-carry contract the harness relies on (§6, §7).
    #[test]
    fn mamba2_decode_equals_prefill() {
        let d = Dims { t: 5, n_head: 3, head_dim: 4, d_state: 3, n_groups: 1 };
        let di = d.d_inner();
        let gd = d.n_groups * d.d_state;
        let mut r = Lcg(0xdead_beef_cafe_1234);
        let x: Vec<f32> = (0..d.t * di).map(|_| r.f(0.5)).collect();
        let b: Vec<f32> = (0..d.t * gd).map(|_| r.f(0.5)).collect();
        let cc: Vec<f32> = (0..d.t * gd).map(|_| r.f(0.5)).collect();
        let dt_eff: Vec<f32> = (0..d.t * d.n_head).map(|_| softplus(r.f(1.0))).collect();
        let a: Vec<f32> = (0..d.n_head).map(|_| -(r.f(0.3) + 0.7).exp()).collect();
        let dd: Vec<f32> = (0..d.n_head).map(|_| r.f(0.5)).collect();

        // Full prefill scan.
        let mut ssm_pf = vec![0.0f32; d.n_head * d.head_dim * d.d_state];
        let y_pf = scan_recurrence(&d, &x, &b, &cc, &dt_eff, &a, &dd, &mut ssm_pf);

        // Token-at-a-time decode, carrying ssm_state forward.
        let mut ssm_dec = vec![0.0f32; d.n_head * d.head_dim * d.d_state];
        let mut y_last = vec![0.0f32; di];
        for t in 0..d.t {
            let d1 = Dims { t: 1, ..copy_dims(&d) };
            let xr = &x[t * di..(t + 1) * di];
            let br = &b[t * gd..(t + 1) * gd];
            let cr = &cc[t * gd..(t + 1) * gd];
            let dtr = &dt_eff[t * d.n_head..(t + 1) * d.n_head];
            y_last = scan_recurrence(&d1, xr, br, cr, dtr, &a, &dd, &mut ssm_dec);
        }
        let err_state = max_abs(&ssm_pf, &ssm_dec);
        let err_y = max_abs(&y_pf[(d.t - 1) * di..], &y_last);
        eprintln!("mamba2 prefill-vs-decode: ssm_state err={err_state:e} last-token err={err_y:e}");
        assert!(err_state < 1e-5, "decode state != prefill state: {err_state:e}");
        assert!(err_y < 1e-5, "decode last-token != prefill: {err_y:e}");
    }

    fn copy_dims(d: &Dims) -> Dims {
        Dims {
            t: d.t,
            n_head: d.n_head,
            head_dim: d.head_dim,
            d_state: d.d_state,
            n_groups: d.n_groups,
        }
    }

    // ---- emit op-sequence + descriptor ---------------------------------------------------------

    /// Synthetic small Nemotron-3 hybrid cfg (structurally faithful: mamba mixer + GQA attn + MoE).
    /// Layer 0 = mamba, 1 = attn, 2 = moe (a minimal one-of-each pattern the block extraction walks).
    fn nemo_ref_cfg() -> NemoCfg {
        NemoCfg {
            layers: 3,
            hidden: 64,
            d_inner: 128,
            n_head: 8,
            head_dim: 16, // d_inner / n_head
            d_state: 16,
            d_conv: 4,
            n_groups: 2,
            attn_heads: 8,
            attn_kv_heads: 2,
            attn_head_dim: 16,
            n_exp: 16,
            top_k: 4,
            shared_exp: 1,
            moe_inter: 96,
            eps: 1e-5,
            kinds: vec![NemoKind::Mamba, NemoKind::Attn, NemoKind::Moe],
        }
    }

    fn block_ops(c: &NemoCfg, block: std::ops::Range<usize>) -> Vec<u16> {
        let (m, _d) = nemotron_build_block(c, 512, 256, block, "nemotron-ref");
        m.progs[0].insts.iter().map(|d| d.op).collect()
    }

    /// Mamba mixer block: input RMSNorm, 3 in_proj GEMVs (z/xBC/dt), the NEW Mamba2Scan, out_proj
    /// GEMV, residual — `act.x` in and out, no embed/tail.
    #[test]
    fn nemotron_mamba_block_sequence() {
        use DevOp::*;
        let c = nemo_ref_cfg();
        assert_eq!(
            block_ops(&c, 0..1),
            vec![RmsNorm, Gemv, Gemv, Gemv, Mamba2Scan, Gemv, Residual]
                .into_iter()
                .map(|o| o as u16)
                .collect::<Vec<_>>(),
            "mamba mixer block sequence"
        );
    }

    /// GQA attention block reuses the existing attn DevOps.
    #[test]
    fn nemotron_attn_block_sequence() {
        use DevOp::*;
        let c = nemo_ref_cfg();
        assert_eq!(
            block_ops(&c, 1..2),
            vec![RmsNorm, GemvQkv, HeadNormRope, FlashDecode, FlashMerge, Gemv, Residual]
                .into_iter()
                .map(|o| o as u16)
                .collect::<Vec<_>>(),
            "gqa attention block sequence"
        );
    }

    /// MoE block reuses the existing MoE DevOps (router split + shared expert + top_k experts +
    /// combine), matching the kimi MoE structure.
    #[test]
    fn nemotron_moe_block_sequence() {
        use DevOp::*;
        let c = nemo_ref_cfg();
        let mut want = vec![RmsNorm, Gemv, MoeRouterTopk, GemvGlu, Gemv];
        for _ in 0..c.top_k {
            want.push(MoeExpertGlu);
            want.push(MoeExpertDown);
        }
        want.push(MoeCombine);
        assert_eq!(
            block_ops(&c, 2..3),
            want.into_iter().map(|o| o as u16).collect::<Vec<_>>(),
            "moe block sequence"
        );
    }

    /// Mamba block descriptor: arch nemotron_h, kind ["mamba2"], Mamba-2 dims, conv+ssm carried
    /// state (NO kv), no attn/MoE dims.
    #[test]
    fn nemotron_mamba_descriptor() {
        let c = nemo_ref_cfg();
        let (_, d) = nemotron_build_block(&c, 512, 256, 0..1, "Nemotron-3");
        assert_eq!(d.arch, "nemotron_h");
        assert_eq!(d.kind, vec!["mamba2"]);
        assert_eq!(d.layer, 0);
        assert_eq!(d.dims.d_inner, Some(128));
        assert_eq!(d.dims.n_head, Some(8));
        assert_eq!(d.dims.head_dim, Some(16));
        assert_eq!(d.dims.d_state, Some(16));
        assert_eq!(d.dims.d_conv, Some(4));
        assert_eq!(d.dims.n_groups, Some(2));
        assert_eq!(d.dims.heads, None, "mamba block has no attn dims");
        assert_eq!(d.dims.n_exp, None, "mamba block has no MoE dims");
        assert_eq!(d.carried_state.len(), 2);
        assert_eq!(d.carried_state[0].role, "conv");
        assert_eq!(d.carried_state[0].layout, "conv");
        assert_eq!(d.carried_state[0].tensors, vec!["mamba.0.conv_state"]);
        assert_eq!(d.carried_state[1].role, "ssm");
        assert_eq!(d.carried_state[1].layout, "ssm_head_major");
        assert_eq!(d.carried_state[1].tensors, vec!["mamba.0.ssm_state"]);
        assert_eq!(d.weights.prefix, "backbone.layers.0.");
        assert!(d.programs.prefill_buckets.is_empty());
        assert_eq!(d.outputs[0].name, "act.xnext", "one (odd) layer -> act.xnext");
    }

    /// Attention block descriptor: kind ["gqa_attn"], GQA dims, kv carried state.
    #[test]
    fn nemotron_attn_descriptor() {
        let c = nemo_ref_cfg();
        let (_, d) = nemotron_build_block(&c, 512, 256, 1..2, "Nemotron-3");
        assert_eq!(d.kind, vec!["gqa_attn"]);
        assert_eq!(d.dims.heads, Some(8));
        assert_eq!(d.dims.kv_heads, Some(2));
        assert_eq!(d.dims.head_dim, Some(16));
        assert_eq!(d.dims.d_inner, None, "attn block has no mamba dims");
        assert_eq!(d.dims.n_exp, None);
        assert_eq!(d.carried_state.len(), 1);
        assert_eq!(d.carried_state[0].role, "kv");
        assert_eq!(d.carried_state[0].tensors, vec!["kv.1.k", "kv.1.v"]);
    }

    /// MoE block descriptor: kind ["moe_ffn"], MoE dims, NO carried state.
    #[test]
    fn nemotron_moe_descriptor() {
        let c = nemo_ref_cfg();
        let (_, d) = nemotron_build_block(&c, 512, 256, 2..3, "Nemotron-3");
        assert_eq!(d.kind, vec!["moe_ffn"]);
        assert_eq!(d.dims.n_exp, Some(16));
        assert_eq!(d.dims.top_k, Some(4));
        assert_eq!(d.dims.shared_exp, Some(1));
        assert_eq!(d.dims.moe_inter, Some(96));
        assert_eq!(d.dims.d_inner, None);
        assert_eq!(d.dims.heads, None);
        assert!(d.carried_state.is_empty(), "MoE block carries no state");
    }

    /// A multi-layer block chains all three layer kinds; kind lists each, carried_state unions the
    /// mamba (conv+ssm) and attn (kv) entries, and the residual ping-pong lands the output in
    /// `act.xnext` after 3 (odd) layers.
    #[test]
    fn nemotron_multi_layer_chains() {
        use DevOp::*;
        let c = nemo_ref_cfg();
        let ops = block_ops(&c, 0..3);
        // mamba(7) + attn(7) + moe(5 + 2*top_k + 1)
        assert_eq!(ops[0], RmsNorm as u16);
        assert_eq!(ops[4], Mamba2Scan as u16, "mamba mixer first");
        assert!(ops.contains(&(FlashDecode as u16)), "attn layer present");
        assert!(ops.contains(&(MoeCombine as u16)), "moe layer present");
        let (_, d) = nemotron_build_block(&c, 512, 256, 0..3, "Nemotron-3");
        assert_eq!(d.kind, vec!["mamba2", "gqa_attn", "moe_ffn"]);
        assert_eq!(d.layer, 0);
        assert_eq!(d.outputs[0].name, "act.xnext", "3 layers (odd) -> act.xnext");
        // conv + ssm (mamba L0) + kv (attn L1); moe contributes none.
        let roles: Vec<&str> = d.carried_state.iter().map(|s| s.role.as_str()).collect();
        assert_eq!(roles, vec!["conv", "ssm", "kv"]);
        // all Mamba-2 dims and attn dims and MoE dims populated.
        assert_eq!(d.dims.d_inner, Some(128));
        assert_eq!(d.dims.kv_heads, Some(2));
        assert_eq!(d.dims.n_exp, Some(16));
    }
}
