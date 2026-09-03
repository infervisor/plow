use super::*;

// ===== Kimi-K3 (`kimi_k3` / text `kimi_linear`) — COMPILER FRONT END ONLY =====================
//
// K3 is a MULTIMODAL wrapper (`KimiK3ForConditionalGeneration`) around a `kimi_linear` text tower.
// The text tower is a HYBRID: 24 of its 93 layers are DeepSeek-style MLA, the other 69 are KDA
// (Kimi Delta Attention — a LINEAR attention with carried recurrent state). Its MoE is LATENT: the
// 896 routed experts read a 3584-wide projection of the hidden state, not the 7168 hidden state,
// and their GEMMs are mxfp4.
//
// Nothing here emits. The job of this section is to get the front end as far as it can honestly
// go — parse every field, resolve the per-layer attention map, cross-check every dimension against
// the safetensors headers actually on disk — and then refuse with an ITEMISED list of what is not
// implemented. A precise refusal is the deliverable; the alternative (reuse `cfg_kimi` because the
// MLA keys happen to have the same spelling) compiles a blob that loads, runs, and is wrong.

/// Per-layer attention implementation in the K3 text tower.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum K3Attn {
    /// DeepSeek-style MLA (q_a/q_b/kv_a_with_mqa/kv_b), 24 layers.
    Mla,
    /// Kimi Delta Attention: linear attention, recurrent state, short convs on q/k/v,
    /// low-rank forget gate. 69 layers. See `docs/kimi-k3-kda.md`.
    Kda,
}

/// Resolved Kimi-K3 text-tower geometry. Every field is REQUIRED — there are no defaults, because
/// a default here is indistinguishable from a correct value at emit time and only shows up as
/// fluent-but-wrong output.
pub(crate) struct K3Cfg {
    layers: u32,
    hidden: u32,
    heads: u32,
    vocab: u32,
    /// Read by the emit path, which is still blocked (see `k3_gaps`). Kept rather than
    /// dropped: re-deriving it later from a different source is how two eps values appear.
    #[allow(dead_code)]
    eps: f32,
    // --- MLA (the 24 full-attention layers) ---
    kv_lora: u32,
    q_lora: u32,
    qk_nope: u32,
    qk_rope: u32,
    v_head: u32,
    /// `mla_use_nope`: MLA carries NO positional encoding (KDA supplies position). The 64 "rope"
    /// dims still exist in the tensors — they are simply never rotated.
    mla_nope: bool,
    /// `mla_use_output_gate`: `self_attn.g_proj` gates the attention output before `o_proj`.
    mla_out_gate: bool,
    /// `rope_theta` if the config carries one. K3's `text_config` carries NONE (consistent with
    /// `mla_use_nope`), and this stays `None` — it is NOT defaulted. `cfg_glm` used to substitute
    /// GLM's 8e6 here; it is now `Option<f64>` too and refuses via `devgen::require_mla_rope`.
    rope_theta: Option<f64>,
    // --- KDA (the 69 linear-attention layers) ---
    kda_heads: u32,
    kda_head_dim: u32,
    kda_conv: u32,
    kda_full_rank_gate: bool,
    kda_gate_lower_bound: f64,
    // --- MoE ---
    n_exp: u32,
    top_k: u32,
    shared_exp: u32,
    moe_inter: u32,
    /// `routed_expert_hidden_size` — the LATENT width the routed-expert GEMMs actually run at.
    moe_latent: u32,
    latent_norm: bool,
    dense_inter: u32,
    first_k_dense: u32,
    route_scale: f32,
    router_sigmoid: bool,
    renormalize: bool,
    n_group: u32,
    topk_group: u32,
    // --- activation ---
    hidden_act: String,
    situ_beta: f64,
    situ_linear_beta: f64,
    // --- residual blocks ---
    attn_res_block: u32,
    // --- quantization ---
    quant_format: String,
    quant_group: u32,
    quant_bits: u32,
    /// Per-layer attention map, 0-BASED and FIRST-CLASS. `attn.len() == layers`.
    attn: Vec<K3Attn>,
    // --- vision (OUT OF SCOPE, refused by name — never silently dropped) ---
    vision: Option<K3Vision>,
}

/// The MoonViT tower + mm_projector this compiler explicitly does NOT implement. Recorded so the
/// refusal can name what it is refusing; never used to emit anything.
struct K3Vision {
    layers: u32,
    hidden: u32,
    projector: String,
}

impl K3Cfg {
    fn n_mla(&self) -> usize {
        self.attn.iter().filter(|&&k| k == K3Attn::Mla).count()
    }
    fn n_kda(&self) -> usize {
        self.attn.iter().filter(|&&k| k == K3Attn::Kda).count()
    }
}

/// Parse `config.json` into a [`K3Cfg`]. Panics (the emitter convention) with a message naming the
/// exact field on anything missing or unexpected.
///
/// Three traps, all of which have a checkpoint-verified answer below:
///  * the geometry lives under `text_config`, not at the root;
///  * the MoE keys are Kimi spellings (`num_experts`, `num_experts_per_token`,
///    `num_shared_experts`), NOT the DeepSeek spellings `cfg_glm` reads;
///  * `linear_attn_config.{full_attn_layers,kda_layers}` are **1-BASED**
///    (`configuration_kimi_k3.py::is_kda_layer` tests `(layer_idx + 1) in kda_layers`).
fn cfg_kimi_k3(dir: &Path) -> K3Cfg {
    let v: Value =
        serde_json::from_slice(&std::fs::read(dir.join("config.json")).expect("config.json"))
            .unwrap();
    k3_cfg_from(&v)
}

/// [`cfg_kimi_k3`] on an already-parsed `config.json`. Split out so the parse rules — above all
/// the 1-based layer lists and the latent-vs-moe_inter choice — are unit-testable without a
/// 618 GB checkpoint on disk.
fn k3_cfg_from(v: &Value) -> K3Cfg {
    // Vision is OUT OF SCOPE and is refused BY NAME (see `kimi_k3_emit`'s SCOPE REFUSAL block and
    // the final panic). It is recorded rather than asserted on here so the text-tower analysis
    // still runs to completion — the report is worth more than an early abort, and the refusal is
    // just as explicit either way. What is NOT acceptable is dropping it silently: a text-only
    // blob for a multimodal checkpoint loads, runs, and is wrong on every image prompt.
    let vision = v
        .get("vision_config")
        .filter(|c| c.is_object())
        .map(|vc| K3Vision {
            layers: vc["vt_num_hidden_layers"].as_u64().unwrap_or(0) as u32,
            hidden: vc["vt_hidden_size"].as_u64().unwrap_or(0) as u32,
            projector: vc["mm_projector_type"].as_str().unwrap_or("?").to_string(),
        });
    let t = &v["text_config"];
    assert!(
        t.is_object(),
        "kimi_k3: config.json has no `text_config` object; the text geometry lives there, not at \
         the root"
    );
    assert_eq!(
        t["model_type"].as_str(),
        Some("kimi_linear"),
        "kimi_k3: text_config.model_type is {:?}, expected \"kimi_linear\"",
        t["model_type"]
    );
    let g = |k: &str| {
        t[k].as_u64()
            .unwrap_or_else(|| panic!("kimi_k3: text_config missing required field {k:?}"))
            as u32
    };
    let gf = |k: &str| {
        t[k].as_f64()
            .unwrap_or_else(|| panic!("kimi_k3: text_config missing required field {k:?}"))
    };
    let gb = |k: &str| {
        t[k].as_bool()
            .unwrap_or_else(|| panic!("kimi_k3: text_config missing required field {k:?}"))
    };
    let layers = g("num_hidden_layers");
    let attn = k3_attn_map(t, layers);

    let lac = &t["linear_attn_config"];
    let q = &t["quantization_config"];
    let qw = &q["config_groups"]["group_0"]["weights"];
    let act = t["hidden_act"]
        .as_str()
        .expect("kimi_k3: text_config missing required field \"hidden_act\"")
        .to_string();

    K3Cfg {
        layers,
        hidden: g("hidden_size"),
        heads: g("num_attention_heads"),
        vocab: g("vocab_size"),
        eps: gf("rms_norm_eps") as f32,
        kv_lora: g("kv_lora_rank"),
        q_lora: g("q_lora_rank"),
        qk_nope: g("qk_nope_head_dim"),
        qk_rope: g("qk_rope_head_dim"),
        v_head: g("v_head_dim"),
        mla_nope: gb("mla_use_nope"),
        mla_out_gate: gb("mla_use_output_gate"),
        // NOT defaulted. Absent means "this model has no RoPE", which is a fact to act on.
        rope_theta: t["rope_theta"].as_f64(),
        kda_heads: lac["num_heads"]
            .as_u64()
            .expect("kimi_k3: linear_attn_config.num_heads") as u32,
        kda_head_dim: lac["head_dim"]
            .as_u64()
            .expect("kimi_k3: linear_attn_config.head_dim") as u32,
        kda_conv: lac["short_conv_kernel_size"]
            .as_u64()
            .expect("kimi_k3: linear_attn_config.short_conv_kernel_size") as u32,
        kda_full_rank_gate: lac["use_full_rank_gate"].as_bool().unwrap_or(false),
        kda_gate_lower_bound: lac["gate_lower_bound"]
            .as_f64()
            .unwrap_or(f64::NEG_INFINITY),
        n_exp: g("num_experts"),
        top_k: g("num_experts_per_token"),
        shared_exp: g("num_shared_experts"),
        moe_inter: g("moe_intermediate_size"),
        moe_latent: g("routed_expert_hidden_size"),
        latent_norm: gb("latent_moe_use_norm"),
        dense_inter: g("intermediate_size"),
        first_k_dense: g("first_k_dense_replace"),
        route_scale: gf("routed_scaling_factor") as f32,
        router_sigmoid: t["moe_router_activation_func"].as_str() == Some("sigmoid"),
        renormalize: gb("moe_renormalize"),
        n_group: g("num_expert_group"),
        topk_group: g("topk_group"),
        situ_beta: t["activation_situ_beta"].as_f64().unwrap_or(f64::NAN),
        situ_linear_beta: t["activation_situ_linear_beta"]
            .as_f64()
            .unwrap_or(f64::NAN),
        hidden_act: act,
        attn_res_block: g("attn_res_block_size"),
        quant_format: q["format"].as_str().unwrap_or("<none>").to_string(),
        quant_group: qw["group_size"].as_u64().unwrap_or(0) as u32,
        quant_bits: qw["num_bits"].as_u64().unwrap_or(0) as u32,
        attn,
        vision,
    }
}

/// Resolve `linear_attn_config.{full_attn_layers,kda_layers}` into a 0-based per-layer map.
///
/// Both lists are read and both are checked: together they must PARTITION `0..layers` — no gap, no
/// overlap, nothing out of range. Deriving one by complement of the other is the §4 bug shape: a
/// truncated list would then silently reclassify layers, and a KDA layer compiled as MLA binds
/// tensor names the checkpoint does not have (`q_a_proj` on a layer that ships `q_proj`).
fn k3_attn_map(t: &Value, layers: u32) -> Vec<K3Attn> {
    let lac = &t["linear_attn_config"];
    assert!(
        lac.is_object(),
        "kimi_k3: text_config has no `linear_attn_config`. Without it there is no way to know \
         which layers are MLA and which are KDA, and guessing the stride mis-binds 69 of 93 layers."
    );
    let list = |k: &str| -> Vec<i64> {
        lac[k]
            .as_array()
            .unwrap_or_else(|| panic!("kimi_k3: linear_attn_config.{k} missing or not an array"))
            .iter()
            .map(|x| {
                x.as_i64()
                    .unwrap_or_else(|| panic!("kimi_k3: linear_attn_config.{k} non-integer entry"))
            })
            .collect()
    };
    let mut out: Vec<Option<K3Attn>> = vec![None; layers as usize];
    for (src, kind) in [
        (list("full_attn_layers"), K3Attn::Mla),
        (list("kda_layers"), K3Attn::Kda),
    ] {
        for one_based in src {
            // 1-BASED -> 0-based, converted exactly once, here.
            let l = one_based - 1;
            assert!(
                (0..layers as i64).contains(&l),
                "kimi_k3: linear_attn_config lists layer {one_based} (1-based; {l} 0-based) but \
                 num_hidden_layers is {layers}"
            );
            assert!(
                out[l as usize].is_none(),
                "kimi_k3: 0-based layer {l} appears in BOTH full_attn_layers and kda_layers"
            );
            out[l as usize] = Some(kind);
        }
    }
    let missing: Vec<usize> = out
        .iter()
        .enumerate()
        .filter(|(_, k)| k.is_none())
        .map(|(i, _)| i)
        .collect();
    assert!(
        missing.is_empty(),
        "kimi_k3: linear_attn_config covers {} of {layers} layers; 0-based layers {:?} are in \
         neither list",
        layers as usize - missing.len(),
        &missing[..missing.len().min(8)]
    );
    out.into_iter().map(|k| k.unwrap()).collect()
}

/// Tensor name -> (dtype, shape) for every `*.safetensors` shard PRESENT in `dir`.
///
/// Deliberately NOT `checkpoint::shard_files`, which panics on an incomplete shard set. K3 is 96
/// shards and a download in progress is the normal case, so this reads whatever has landed and
/// reports the count. **A tensor's absence proves nothing** — every caller below must only ever
/// use this to CONTRADICT the config, never to conclude something does not exist.
fn k3_shard_headers(
    dir: &Path,
) -> (
    std::collections::BTreeMap<String, (String, Vec<i64>)>,
    u32,
    u32,
) {
    use std::io::Read;
    let mut out = std::collections::BTreeMap::new();
    let (mut have, mut total) = (0u32, 0u32);
    let Ok(rd) = std::fs::read_dir(dir) else {
        return (out, 0, 0);
    };
    let mut files: Vec<std::path::PathBuf> = Vec::new();
    for ent in rd.flatten() {
        let fname = ent.file_name();
        let Some(f) = fname.to_str() else { continue };
        let base = f
            .strip_suffix(".partial.safetensors")
            .or_else(|| f.strip_suffix(".safetensors"));
        let Some(base) = base else { continue };
        if let Some((_, n)) = base.rsplit_once("-of-") {
            total = total.max(n.parse::<u32>().unwrap_or(0));
        }
        files.push(ent.path());
    }
    files.sort();
    for p in &files {
        let Ok(mut f) = std::fs::File::open(p) else {
            continue;
        };
        let mut len8 = [0u8; 8];
        if f.read_exact(&mut len8).is_err() {
            continue;
        }
        let hlen = u64::from_le_bytes(len8);
        if hlen == 0 || hlen > 256 * 1024 * 1024 {
            continue;
        }
        let mut hbuf = vec![0u8; hlen as usize];
        if f.read_exact(&mut hbuf).is_err() {
            continue; // still downloading: header not fully written yet
        }
        let Ok(hdr) = serde_json::from_slice::<Value>(&hbuf) else {
            continue;
        };
        let Some(obj) = hdr.as_object() else { continue };
        have += 1;
        for (k, val) in obj {
            if k == "__metadata__" {
                continue;
            }
            let shape: Vec<i64> = val["shape"]
                .as_array()
                .map(|a| a.iter().filter_map(|x| x.as_i64()).collect())
                .unwrap_or_default();
            out.insert(
                k.clone(),
                (val["dtype"].as_str().unwrap_or("?").to_string(), shape),
            );
        }
    }
    (out, have, total.max(have))
}

/// Cross-check every config dimension against the shard headers on disk. **Tensors win.**
///
/// This exists because of the GLM-5.2 lesson (§ knob-contract): `AutoConfig` reported
/// `qk_rope_head_dim=192` where the tensors said 64 and it cost a day. Returns one line per
/// disagreement; an empty vector means every dim the checkpoint can speak to agrees with the config.
/// Tensors that have not downloaded yet are simply not checked.
fn k3_config_vs_tensors(
    c: &K3Cfg,
    h: &std::collections::BTreeMap<String, (String, Vec<i64>)>,
) -> Vec<String> {
    let mut errs = Vec::new();
    let mut check = |name: String, want: Vec<i64>| {
        if let Some((_, got)) = h.get(&name) {
            if *got != want {
                errs.push(format!(
                    "{name}: config implies {want:?}, tensor is {got:?}"
                ));
            }
        }
    };
    let (hd, nh) = (c.hidden as i64, c.heads as i64);
    // Pick the first MLA and first KDA layer that actually exist on disk.
    let mla_l = c.attn.iter().position(|&k| k == K3Attn::Mla);
    let kda_l = c.attn.iter().position(|&k| k == K3Attn::Kda);
    let p = "language_model.model.layers";
    if let Some(l) = mla_l {
        let a = format!("{p}.{l}.self_attn");
        check(format!("{a}.q_a_proj.weight"), vec![c.q_lora as i64, hd]);
        check(
            format!("{a}.q_b_proj.weight"),
            vec![nh * (c.qk_nope + c.qk_rope) as i64, c.q_lora as i64],
        );
        check(
            format!("{a}.kv_a_proj_with_mqa.weight"),
            vec![(c.kv_lora + c.qk_rope) as i64, hd],
        );
        check(
            format!("{a}.kv_b_proj.weight"),
            vec![nh * (c.qk_nope + c.v_head) as i64, c.kv_lora as i64],
        );
        check(format!("{a}.o_proj.weight"), vec![hd, nh * c.v_head as i64]);
        if c.mla_out_gate {
            check(format!("{a}.g_proj.weight"), vec![nh * c.v_head as i64, hd]);
        }
    }
    if let Some(l) = kda_l {
        let a = format!("{p}.{l}.self_attn");
        let w = (c.kda_heads * c.kda_head_dim) as i64;
        for proj in ["q_proj", "k_proj", "v_proj", "g_proj"] {
            check(format!("{a}.{proj}.weight"), vec![w, hd]);
        }
        for cv in ["q_conv1d", "k_conv1d", "v_conv1d"] {
            check(format!("{a}.{cv}.weight"), vec![w, 1, c.kda_conv as i64]);
        }
        check(format!("{a}.b_proj.weight"), vec![c.kda_heads as i64, hd]);
        check(format!("{a}.o_proj.weight"), vec![hd, w]);
    }
    // MoE on the first MoE layer present.
    if let Some(l) = (c.first_k_dense..c.layers).next() {
        let m = format!("{p}.{l}.block_sparse_moe");
        check(format!("{m}.gate.weight"), vec![c.n_exp as i64, hd]);
        check(
            format!("{m}.gate.e_score_correction_bias"),
            vec![c.n_exp as i64],
        );
        check(
            format!("{m}.routed_expert_down_proj.weight"),
            vec![c.moe_latent as i64, hd],
        );
        check(
            format!("{m}.routed_expert_up_proj.weight"),
            vec![hd, c.moe_latent as i64],
        );
        check(
            format!("{m}.routed_expert_norm.weight"),
            vec![c.moe_latent as i64],
        );
        let sh = (c.shared_exp * c.moe_inter) as i64;
        check(format!("{m}.shared_experts.gate_proj.weight"), vec![sh, hd]);
        check(format!("{m}.shared_experts.up_proj.weight"), vec![sh, hd]);
        check(format!("{m}.shared_experts.down_proj.weight"), vec![hd, sh]);
        // mxfp4 routed expert 0. `weight_packed` is [N, K/2] (2 fp4 per byte) and
        // `weight_scale` is [N, K/group] (one E8M0 byte per group) — the SAME layout
        // DevOp::GemvMxfp4 documents (crates/packet/src/dev.rs:622). K is the LATENT
        // width, not moe_inter: that is the load-bearing check in this whole function.
        let (li, lo) = (c.moe_latent as i64, c.moe_inter as i64);
        let grp = c.quant_group.max(1) as i64;
        for w13 in ["w1", "w3"] {
            check(
                format!("{m}.experts.0.{w13}.weight_packed"),
                vec![lo, li / 2],
            );
            check(
                format!("{m}.experts.0.{w13}.weight_scale"),
                vec![lo, li / grp],
            );
        }
        check(format!("{m}.experts.0.w2.weight_packed"), vec![li, lo / 2]);
        check(format!("{m}.experts.0.w2.weight_scale"), vec![li, lo / grp]);
    }
    if c.first_k_dense > 0 {
        let m = format!("{p}.0.mlp");
        check(
            format!("{m}.gate_proj.weight"),
            vec![c.dense_inter as i64, hd],
        );
        check(
            format!("{m}.up_proj.weight"),
            vec![c.dense_inter as i64, hd],
        );
        check(
            format!("{m}.down_proj.weight"),
            vec![hd, c.dense_inter as i64],
        );
    }
    errs
}

/// One K3 capability: what it is, why it blocks, and where the fix goes — plus, once it lands,
/// the evidence that it did.
///
/// # Why closed items STAY in this list
///
/// This report's own preamble says a gap list on its own "invites the next agent to rebuild
/// machinery that exists — the mirror image of §4's *an arm exists and nothing routes to it*".
/// That happened. Between `3f64b3c` and `6603cf7` SIX of these entries were implemented and
/// validated against real-weight oracles on gfx950 — KDA, `situ`, AttnRes, LatentMoE, the MLA
/// output gate and NoPE — and the list went on printing all six as unimplemented blockers. A
/// reader, human or agent, who trusts it is told to write seven opcodes that already dispatch
/// (88, 89, 102, 103, 104, 105, 106, all of them in `GFX950_DISPATCHED`), and is NOT told that
/// the one thing actually standing between this checkpoint and a token is the model-level
/// assembly. The report said "8 unimplemented capabilities"; the true count was 2.
///
/// Deleting a closed entry is the opposite failure: the next agent re-derives whether it was ever
/// needed. So entries are RETIRED, not removed — `done` carries the commit and the measured
/// residual, and only `done.is_none()` entries count as blockers.
struct K3Gap {
    what: &'static str,
    scope: String,
    why: String,
    fix: &'static str,
    /// `Some(evidence)` once this landed and passed a real-weight numeric gate on hardware.
    /// The string is printed verbatim in the CLOSED section and is the reason not to rebuild it.
    done: Option<&'static str>,
}

/// The ranked missing-capability list for Kimi-K3, blocker first.
///
/// Ordering rule: a gap that blocks EVERY layer outranks one that blocks a subset, and a gap whose
/// SEMANTICS are unknown outranks one that is only unwired — you cannot schedule work you cannot
/// specify.
fn k3_gaps(c: &K3Cfg) -> Vec<K3Gap> {
    let mut g = Vec::new();
    g.push(K3Gap {
        what: "KDA (Kimi Delta Attention) linear attention",
        scope: format!("{}/{} layers", c.n_kda(), c.layers),
        why: format!(
            "linear attention with CARRIED RECURRENT STATE ([{}h x {}d x {}d] per layer), short \
             depthwise convs (k={}) on q/k/v, a {} forget gate (f_a_proj/f_b_proj) and A_log/dt_bias. \
             SEMANTICS ARE SPECIFIED by `docs/kimi-k3-kda.md`, and they are now IMPLEMENTED in the \
             dataflow form that doc's §7.2 `KdaScan` proposal was overruled in favour of: the \
             recurrent state is a DECLARED HBM TENSOR with counter-gated tile dependencies, not a \
             monolithic state-carrying kernel.",
            c.kda_heads,
            c.kda_head_dim,
            c.kda_head_dim,
            c.kda_conv,
            if c.kda_full_rank_gate { "full-rank" } else { "low-rank" },
        ),
        fix: "DONE — crates/devgen/src/kda.rs (declare_kda_weights/declare_kda_state/\
              emit_kda_layer) + runtime/amd/op_kda.h. What is NOT done is the CALL: \
              `emit_kda_layer` is reached by nothing outside kda.rs's own unit tests, because \
              there is no model-level K3 emitter to call it (see the full-model-emit gap).",
        done: Some(
            "3f64b3c. FOUR opcodes, not one, all in GFX950_DISPATCHED and all reached by the \
             emitter: KdaConv=88, KdaGate=89, KdaStateStep=102, KdaGatedNorm=103. Real layer-0 \
             weights, 16 packets on a leased gfx950, T=1 AND T=4: conv+SiLU ~2.4e-03, gate \
             2.0e-04, beta 2.5e-04, STATE (f32, V-first) 1.4e-04, block out 8.1e-04. The state \
             row is the load-bearing one — against the TRANSPOSED reading of the same reference \
             it is 1.408e+00, 10100x larger, and both readings have identical norms, so no \
             magnitude check would have caught a transpose. REGISTER COST ZERO: decode stayed \
             248 VGPR / occ 2 / spill 0 (the predicted 32 VGPR/lane was the cost of one workgroup \
             per head, not of KDA — one WAVE owns one column, so the state is 2 f32/lane). \
             `Mamba2Scan(90)`, the dead-opcode cautionary tale this was written against, is still \
             dead and still has no AMD arm.",
        ),
    });
    if c.hidden_act != "silu" && c.hidden_act != "gelu_pytorch_tanh" && c.hidden_act != "gelu_tanh"
    {
        g.push(K3Gap {
            what: "`situ` activation",
            scope: "every FFN: dense layer 0, 2 shared experts, 896 routed experts, all 93 layers"
                .into(),
            why: format!(
                "hidden_act = {:?} (beta {}, linear_beta {}). CLOSED FOR DECODE, OPEN FOR PREFILL. \
                 The original diagnosis — \"plow's activation operand is ONE BIT\" — was right \
                 about the constraint and wrong about the fix: situ transforms the UP branch as \
                 well as the gate, `beta*tanh(g/beta)*sigmoid(g) * lbeta*tanh(u/lbeta)`, so the \
                 EXPRESSION SHAPE changes from `act(g)*u` to `A(g)*B(u)` and a third `act` code \
                 alone would have left `up` un-clipped — a small error at |u| < {} that grows with \
                 the tail, i.e. plausible output and the wrong model. WHAT IS STILL OPEN: the two \
                 GROUPED PREFILL GLU epilogues (runtime/amd/op_moe.h:1285, :1584) were not \
                 converted to the pair form, so `moe_act` returns NaN for code 2 rather than \
                 silently computing gelu_tanh(g)*u. Prefill for a K3 MoE layer is therefore \
                 REFUSED-BY-NaN, not supported.",
                c.hidden_act, c.situ_beta, c.situ_linear_beta, c.situ_linear_beta
            ),
            fix: "DONE for decode. OPEN: runtime/amd/op_moe.h:1285 and :1584 (convert the grouped \
                  prefill epilogues to the pair form `moe_glu(g, u, act, beta, lbeta)`), which is \
                  a precondition for any K3 PREFILL program.",
            done: Some(
                "50d9ed5 (dense/shared) + the routed-expert half. NOT a third `act` code: \
                 PLOW_DOP_SITU_GLU = 105 for the dense and shared FFNs, and PLOW_MOE_ACT_SITU = 2 \
                 with a PAIR-form `moe_glu` inside the 896 routed experts. The two betas ride in \
                 `f0`/`f1`, which were FREE on every GLU-family op, so no `i` slot moved and every \
                 pre-K3 packet is byte-identical. Measured on real weights: dense situ act \
                 3.177e-03 (rung 1), expert situ GLU 3.553e-03 (rung 2), 1.815e-03 (rung 3). \
                 Register cost ZERO across all four objects. `moe_act` returning NaN for code 2 is \
                 deliberate — this interpreter's dispatch `default:` is a silent NOP and there is \
                 no device trap, so NaN is the loudest primitive available for the two epilogues \
                 that were not converted.",
            ),
        });
    }
    g.push(K3Gap {
        what: "residual-ATTENTION blocks (`attn_res_block_size`) — not a residual add at all",
        scope: format!(
            "all {} layers, TWICE each (post-attention and post-MLP), plus once at the model output",
            c.layers
        ),
        why: format!(
            "`_apply_attn_res` (modeling_kimi_linear.py:1075) replaces `x = x + f(x)` with a SOFTMAX \
             over up to {} candidates: the running prefix sum plus one snapshot per completed \
             {}-layer block. It RMS-normalises each candidate, scores it against \
             `norm.weight * proj.weight` and takes a probability-weighted mixture. So every layer \
             ships `self_attention_res_norm` [{h}] + `self_attention_res_proj` [1,{h}] and \
             `mlp_res_norm` [{h}] + `mlp_res_proj` [1,{h}], the model ships an `output_attn_res_*` \
             pair, and a new block snapshot is pushed when `layer_idx % {} == 0`. \
             `score_weight = norm.weight * proj.weight` is a constant [{h}] vector, so the two \
             tensors FOLD INTO ONE at weight-prep time and neither factor reaches the device.\n\
             THE DETECTABILITY FINDING, which is the reason this entry is worth reading even \
             though the op is done: at a SNAPSHOT layer (l % {} == 0, i.e. {} of {}) the block \
             output is `attn + ffn` and a plain-residual wiring differs by 1.0. At EVERY OTHER \
             layer `prefix = prefix_in + attn`, so the block output is `prefix_in + attn + ffn` — \
             EXACTLY what a plain residual produces, measured at 3.0e-03 on real layer-1 weights \
             against 8.1e-01 at the AttnRes outputs themselves. **A block-output-only gate does \
             not see AttnRes at {} of {} layers.** Any future K3 gate must score the two AttnRes \
             outputs, not the block output.",
            c.layers / c.attn_res_block.max(1) + 1,
            c.attn_res_block,
            c.attn_res_block,
            c.attn_res_block,
            c.layers.div_ceil(c.attn_res_block.max(1)),
            c.layers,
            c.layers - c.layers.div_ceil(c.attn_res_block.max(1)),
            c.layers,
            h = c.hidden
        ),
        fix: "DONE — runtime/amd/op_k3.h (op 104) + crates/devgen/src/k3.rs emit_attn_res. What is \
              NOT done is the model-level plumbing: the `block_residual` ring (<=8 live H-wide \
              snapshots) as CARRIED STATE across the layer loop, which belongs to the full-model \
              emit gap, not here.",
        done: Some(
            "50d9ed5. PLOW_DOP_ATTN_RES = 104, ONE packet (three packets x 186/token would be \
             3.3 ms of pure protocol). Real weights: AttnRes(attn) 0.000e+00 and AttnRes(mlp) \
             1.109e-03 at rung 2; 1.000e-03 at rung 1; EXACTLY 0 at rung 3. Controls at the \
             sub-layer inputs, which is where they have to be: h_a vs a plain residual 8.04e-01, \
             h2 vs plain 7.70e-01. Three things the gate pinned that a code read would not: \
             `score_weight` folds at prep time; the mix is over the RAW rows v, not the normalised \
             k (the natural misreading — right shape, wrong per-row magnitude); and \
             `variance = mean(x^2)` is RMSNorm's, not mean-centred. KNOWN COST, UNMEASURED: \
             AttnRes is ONE WORKGROUP PER TOKEN (blocks = 1 of 256 at T=1) because both reductions \
             span the full H-wide row and the softmax couples the rows. 186 invocations/token on \
             1 CU. The batched form is the fix and it is not written.",
        ),
    });
    if c.moe_latent > 0 && c.moe_latent != c.hidden {
        g.push(K3Gap {
            what: "LATENT MoE (routed experts do not read the hidden state)",
            scope: format!("{} MoE layers", c.layers - c.first_k_dense),
            why: format!(
                "resolved order, from modeling_kimi_linear.py:815-837 — the ROUTER scores the \
                 HIDDEN state ({}), then `routed_expert_down_proj` projects hidden -> latent {}, \
                 every routed expert runs at K={}, the gated expert sum is RMS-normed by \
                 `routed_expert_norm` [{}] (latent_moe_use_norm={}) and only then does \
                 `routed_expert_up_proj` [{},{}] return to hidden. The shared experts read the \
                 ORIGINAL hidden and are added AFTER the up-projection. THE KERNELS NEEDED \
                 NOTHING: H is a runtime operand and the scale-row arithmetic needs only 128- and \
                 32-divisibility, which {}/128 and {}/32 satisfy exactly. What is STILL OPEN is \
                 the DECLARE: `declare_glm` sizes every expert weight with K = hidden, wrong here \
                 by {}/{} = 2x, and the combine accumulator must run at latent rather than hidden \
                 width (the four decode `MoeCombine` `d.i[0] = h` sites and the two prefill \
                 combine sites, which perf-data/archive/k3/kimi-k3-kernel-gap.md §5e omits from its list).",
                c.hidden,
                c.moe_latent,
                c.moe_latent,
                c.moe_latent,
                c.latent_norm,
                c.hidden,
                c.moe_latent,
                c.moe_latent,
                c.moe_latent,
                c.hidden,
                c.moe_latent,
            ),
            fix: "crates/devgen/src/mla.rs declare_glm (expert weight/scale sizing keyed on the \
                  latent width) and the MoE emit (down/norm/up around the expert loop). The GRAPH \
                  is proven (see below); this is the width plumbing, and it belongs with the \
                  full-model emit.",
            done: Some(
                "50d9ed5, GRAPH ONLY — the kernels were already sufficient. Validated on real \
                 layer-1 weights, top-16 of 896, on real mxfp4 bytes: latent down 2.158e-03, \
                 expert situ GLU 3.553e-03, MoeCombine(no residual) 3.378e-03, latent RMSNorm \
                 3.673e-03, latent up 4.123e-03, shared expert 3.993e-03. One kernel line \
                 changed: `d_moe_combine`'s `residual` is now OPTIONAL (op_moe.h:819 decode, \
                 :1689 prefill) — it was an unconditional null deref, and a latent-width combine \
                 has no hidden-width residual to add. TWO OPERAND FACTS the gate pins that a code \
                 read would not: the shared expert reads the PRE-DOWN hidden `h3` (feeding it the \
                 latent fails loudly on width; feeding it `h2` would fail QUIETLY), and the gate \
                 weight multiplies inside the DOWN kernel, not in the combine.",
            ),
        });
    }
    if c.top_k > crate::MOE_MAX_TOPK {
        g.push(K3Gap {
            what: "top-k beyond PLOW_MOE_MAX_TOPK",
            scope: format!("top-{} routing on {} MoE layers", c.top_k, c.layers - c.first_k_dense),
            why: format!(
                "`#define PLOW_MOE_MAX_TOPK` (runtime/amd/op_moe.h) sizes both routers\' winner/gate \
                 arrays and the `wl` LDS carve the rank pass writes into. This checkpoint routes \
                 top-{}, past the current bound of {}. The emit refuses (devgen::require_moe_topk) \
                 rather than letting the kernel truncate: slots above the bound are never written, \
                 every expert body loops to the packet\'s unbounded top_k operand and reads them as \
                 uninitialised scratch, and the renormalisation denominator covers only the kept \
                 gates. Raise both constants together — a drift test enforces the pair.",
                c.top_k, crate::MOE_MAX_TOPK
            ),
            fix: "runtime/amd/op_moe.h:57 (raise the bound, re-check the LDS carve at :299), and \
                  turn the two silent clamps at :135/:314 into a hard failure so the next model \
                  past the bound is loud instead of wrong.",
            done: None,
        });
    }
    if c.mla_out_gate {
        g.push(K3Gap {
            what: "MLA output gate (`mla_use_output_gate`)",
            scope: format!("{} MLA layers", c.n_mla()),
            why: format!(
                "`self_attn.g_proj.weight` [{}, {}] = [heads*v_head_dim, hidden] gates the \
                 attention output before o_proj; plow's MLA chain was flash -> OUvFold -> o_proj \
                 with nothing in between. Now expressed as its own opcode rather than folded into \
                 `MlaMergeFold`'s epilogue: the fold is a REDUCTION over KV splits and the gate is \
                 a per-element multiply on its RESULT, so folding it in would have applied the \
                 sigmoid once per split.",
                c.heads * c.v_head,
                c.hidden
            ),
            fix: "DONE — runtime/amd/op_k3.h (op 106) + crates/devgen/src/k3.rs \
                  emit_mla_out_gate. The CALL site is the model-level MLA emit, which does not \
                  exist yet (see the full-model emit gap).",
            done: Some(
                "6603cf7 (rung 3). PLOW_DOP_MLA_OUT_GATE = 106. Real layer-3 weights: MLA OUTPUT \
                 GATE 3.468e-05, block output 7.324e-04, with the control `gated vs ungated \
                 attention` at 5.17e-01 — i.e. the gate is not a rounding-level effect and a \
                 missing one would not have hidden in the block output.",
            ),
        });
    }
    if c.mla_nope || c.rope_theta.is_none() {
        g.push(K3Gap {
            what: "MLA with NO positional encoding (`mla_use_nope`)",
            scope: format!("{} MLA layers", c.n_mla()),
            why: format!(
                "mla_use_nope={} and text_config carries NO `rope_theta` at all — KDA supplies \
                 position, so the {dr} decoupled dims exist in q_b/kv_a but are never rotated \
                 (modeling_kimi_linear.py: `self.rotary_emb = None`, `assert self.use_nope`; \
                 q_rot/k_rot are split off and concatenated back UNCHANGED, i.e. they are extra \
                 CONTENT dims of the {}-wide key).\n\
                 THE SILENT DEFAULT IS CLOSED: `cfg_glm` no longer reads `rope_theta` as \
                 `.unwrap_or(8_000_000.0)`; it is `Option<f64>`, both config spellings are tried, \
                 and `devgen::require_mla_rope` REFUSES a NoPE checkpoint at parse time instead of \
                 substituting GLM's theta. (That default was also load-bearing for GLM-5.2 itself, \
                 whose config has no top-level `rope_theta` — the key moved to \
                 `rope_parameters.rope_theta` in transformers 5.x.)\n\
                 WHAT REMAINS IS THE EMIT, and it is not a deletion. `emit_glm_mla` has two \
                 HeadNormRope ops; the k-side one (mla.rs, `d.t[0] = n.krot[slot]`) is ALSO THE \
                 ONLY WRITER OF THE `kv.{{l}}.krot` CACHE ROW. Drop it and the rope half of every \
                 cached key is never written while FlashMlaDecode keeps reading it at i[5] — \
                 uninitialised memory that grows with context and never faults. WORSE, that op is \
                 also how the RUNTIME FINDS the per-layer KV-row writer: `plowrt::exec::amd`'s \
                 kv_row_writer classifier and runtime/tests/glm52_decode.c:419 both SCAN the \
                 instruction stream for a HeadNormRope whose t[0] is a `kv.*.krot` tensor and \
                 patch its out_row to the current position every step. Delete it and the scan \
                 simply finds fewer layers — no error, no count check. So a NoPE MLA needs the \
                 WRITE KEPT and the ROTATION removed, not the op removed; \
                 perf-data/archive/k3/kimi-k3-kernel-gap.md 8c and item #2 (\"a removal, effort XS\") are \
                 wrong on this point. The KV layout does NOT change: krot stays [ctx][{dr}] and \
                 holds the raw, unrotated k_rot.",
                c.mla_nope,
                c.qk_nope + c.qk_rope,
                dr = c.qk_rope,
            ),
            fix: "TECHNIQUE PROVEN, RUST EMIT NOT WRITTEN. `crates/devgen/src/k3.rs \
                  k3_nope_rope_pair` builds the identity table and a unit test checks it BITWISE; \
                  what is missing is `emit_glm_mla` / `emit_glm_mla_prefill` selecting it off \
                  `rope_theta == None` and keeping both HeadNormRope emits. NOTE FOR ANYONE TOLD \
                  \"just open require_mla_rope for K3\": that gate is on the `cfg_glm` path and \
                  the K3 config parse (`k3_cfg_from`) NEVER CALLS IT — it is not what blocks the \
                  93-layer emit, and opening it changes nothing. The blocker is the model-level \
                  emitter below.",
            done: Some(
                "6603cf7, TECHNIQUE ONLY — proven in the rung-3 C harness, not in devgen. NoPE is \
                 done with an IDENTITY cos=1/sin=0 table (both exact in bf16, so HeadNormRope is a \
                 bit-exact row copy), keeping BOTH HeadNormRope emits so the krot cache write and \
                 the runtime's kv-row-writer scan both survive. Real layer-3 weights: absorbed \
                 q_nope 1.724e-07, FLASH_MLA+MERGE_FOLD 1.069e-03, and both KV writes EXACTLY 0. \
                 THE CONTROL IS THE PART WORTH INHERITING: the first version rotated q and every \
                 cached k at the SAME position and measured 1.2e-07, i.e. 'RoPE is harmless here' \
                 — a common rotation is ORTHOGONAL and preserves every dot product exactly. RoPE \
                 is RELATIVE: key t must be rotated by t, query by qpos. Corrected control \
                 2.459e-01. A control that proves nothing is worse than no control.",
            ),
        });
    }
    g.push(K3Gap {
        what: "full-model emit for a hybrid MLA arch — THE ONE REMAINING BLOCKER",
        scope: "the whole blob".into(),
        why:
            "EVERY OP K3 NEEDS NOW EXISTS AND PASSES A REAL-WEIGHT GATE (see CLOSED, above). What \
              does not exist is anything that CALLS them together. Concretely, and this is the \
              honest state of the tree rather than a plan:\n\
              * `crates/devgen/src/k3.rs` and `crates/devgen/src/kda.rs` are reached by NOTHING \
                outside their own `#[cfg(test)]` modules. `emit_kda_layer` emits a whole KDA \
                mixer; `emit_attn_res`/`emit_situ_glu`/`emit_mla_out_gate`/`emit_k3_block_out` \
                emit one packet each. No function composes them into even ONE complete layer, and \
                there is no loop over layers anywhere.\n\
              * the three rung gates build their instruction streams BY HAND IN C \
                (`runtime/tests/k3_{block,moe_block,mla_block}_gfx950_test.c`, a private \
                `emitop()` each), against fixtures pinned to a single `K3_LAYER`. So the C \
                harnesses and the devgen modules are two independent transcriptions of the same \
                graph with nothing tying them together — a drift hazard that only a shared emit \
                closes.\n\
              * `glm_emit_full` cannot be reused as-is: it assumes a UNIFORM MLA layer and a PLAIN \
                RESIDUAL ADD, and K3 breaks both. It needs the per-layer attention map to select \
                layer L's ops AND its carried state (a KDA recurrent state + 3 conv states on 69 \
                layers, a ckv/krot KV ring on 24), plus the `block_residual` snapshot ring.\n\
              * there is no embed / final-norm / lm_head / argmax tail for K3 in ANY form: all \
                three C gates are block-only, `act.x` in and a residual out.\n\
              * and there is a SECOND, independent refusal outside devgen: \
                `crates/plowc/src/hf_config.rs` `build_full_model_plan` asserts \
                `arch != HfArch::KimiK3`, locked by `test_kimi_k3_has_no_full_model_plan`. Both \
                have to open.\n\
              THERE IS ALSO NO TRUNCATION KNOB. GLM's `GLM_NLAYERS` (mla.rs, `glm_emit_full`) is \
              what makes a cheap iteration loop possible — a truncated model loads in seconds \
              instead of the 4-minute 183 GiB/rank full load. K3 has no equivalent and cannot \
              have one until there is a loop to truncate. When it is written: layers 0..3 is the \
              minimum honest span, because 0/1/2 are KDA and 3 is the first MLA, so anything \
              shorter is not testing the hybrid at all."
                .into(),
        fix: "crates/devgen/src/lib.rs (dispatch: stop routing kimi_k3 unconditionally into \
              `kimi_k3_emit`), crates/plowc/src/hf_config.rs (the second refusal), and a new \
              `k3_main`/`k3_emit_full` in crates/devgen/src/mla.rs or a k3 module: a declare keyed \
              on the per-layer attention map, the layer loop composing kda.rs + k3.rs + the MLA \
              emit, a K3_NLAYERS truncation knob, and the embed/tail.",
        done: None,
    });
    g.push(K3Gap {
        what: "MoE sub-namespace + expert-name template",
        scope: "every expert tensor on 92 MoE layers".into(),
        why: "PARTLY CLOSED. The wrapper prefix itself is no longer a gap in either half:\n\
              * emit — `GlmCfg::prefix` is now cfg data (mirroring `Cfg::prefix` on the Gemma \
                path) and `declare_glm` builds every name from it, so a tower spelled \
                `language_model.model.layers.{L}.…` needs a field, not a patch;\n\
              * bind — the loaders no longer allowlist weight prefixes. `packet::names::\
                is_checkpoint_weight` classifies by EXCLUSION of the compiler's own namespaces \
                (`act.`/`in.`/`kv.`/`moe.` + the host-filled pointer tables), so an unknown name \
                is demanded of the checkpoint and a missing one is `MISSING WEIGHT: <name>`. \
                Under the old `starts_with(\"model.\")` all 497 052 of this checkpoint's \
                language-tower tensors — none of which starts with `model.` — would have been \
                allocated, never uploaded, zero-filled and decoded from.\n\
              What is left is BELOW the prefix: the MoE block is `block_sparse_moe.…` with \
              experts `experts.{e}.w1|w2|w3` (Mixtral naming: w1=gate, w2=down, w3=up) and \
              mxfp4 `weight_packed`/`weight_scale`, where `declare_glm` and \
              `bind_packed_experts` both spell `mlp.experts.{e}.{gate,up,down}_proj.weight`.\n\
              A THIRD SITE, not previously recorded here and the dangerous one: the TP shard \
              classifier `crates/plowrt/src/asset/shard.rs` keys on projection SUBSTRINGS — `COL` \
              holds \"gate_proj.weight\"/\"up_proj.weight\", `ROW` holds \
              \"o_proj.weight\"/\"down_proj.weight\", matched with `name.contains(s)`. Mixtral \
              `w1`/`w2`/`w3` match NEITHER list, so every routed-expert tensor would fall through \
              to `Shard::Replicated` — no error, no missing weight, just every rank holding the \
              whole expert and column-parallel work done redundantly against a row-parallel \
              layout. It is a substring default, so it fails by SILENCE in exactly the way this \
              report exists to prevent.\n\
              A FOURTH, for the same template: there is NO mxfp4 expert-bind path in the AMD \
              runtime at all. `bind_packed_experts` binds `.weight` + `.weight_scale_inv` \
              (block-fp8) only; K3 ships `weight_packed` + `weight_scale`. The decode KERNEL arm \
              exists and is validated — what is missing is the host-side bind."
            .into(),
        fix: "crates/devgen/src/mla.rs declare_glm (expert-name template as cfg data, next to \
              `GlmCfg::prefix`), crates/plowrt/src/exec/amd.rs bind_packed_experts (same \
              template, read from the packet rather than hardcoded, plus a weight_packed/\
              weight_scale arm), and crates/plowrt/src/asset/shard.rs (an axis tag per \
              projection, or the same template — NOT another substring literal).",
        done: None,
    });
    g
}

/// `--emit devblob` on a Kimi-K3 checkpoint. Parses and validates everything the front end can,
/// prints the state of the checkpoint and the itemised missing-capability report, then aborts.
///
/// This function never returns: there is no correct blob to emit. It exists so the failure is a
/// specific, accurate statement of what is not implemented rather than an `Option::unwrap` panic
/// three crates deep (which is what `kimi_k3` produced before: `crates/devgen/src/config.rs:93`,
/// because the `text_config` probe routed it into the Gemma-4 parser).
/// Emit the full K3 decode blob. `K3_FULL=1` selects this; the default stays
/// the capability report in [`kimi_k3_emit`], which is still the honest answer
/// for anyone who has not read what is missing.
///
/// `K3_NLAYERS` truncates, and it is what makes iteration affordable — the same
/// role `GLM_NLAYERS` plays. **0..3 is the minimum honest span**: 0/1/2 are KDA
/// and 3 is the first MLA, so anything shorter does not exercise the hybrid at
/// all. Truncation shrinks the tensor table, so a short model loads in seconds
/// instead of paying the full-checkpoint load.
///
/// What this does NOT yet do, and what will therefore fail at LOAD rather than
/// here: the host-side mxfp4 expert bind (`bind_packed_experts` knows
/// `.weight` + `.weight_scale_inv`, K3 ships `weight_packed` + `weight_scale`)
/// and the Mixtral `w1/w2/w3` expert-name template. Both fail loudly with a
/// missing weight, which is the right failure — but they are why this is gated
/// rather than default.
#[allow(clippy::too_many_arguments)]
pub(crate) fn k3_emit_full(
    dir: &Path,
    ctx: u32,
    out: &str,
    n_cu: u32,
    tp: u32,
    rope_gen: bool,
    target: &str,
    verify: Option<&crate::VerifyHook>,
    l2_layout: Option<packet::devbuild::L2Layout>,
) {
    let c = cfg_kimi_k3(dir);
    let pf = k3_prefill_buckets(ctx);
    let mut m = k3_build_model(dir, ctx, n_cu, tp, &pf, l2_layout);
    k3_ablate_bodies(&mut m);
    // Leave the position tables as GENERATED tensors unless asked to bake them.
    // The runtime materialises them at load (`exec/amd.rs` `g.generate()`), and
    // `DevBlob::parse` refuses a gen tensor it cannot produce, so nothing is
    // taken on trust. Baking is `--no-rope-gen`.
    //
    // It matters more here than anywhere else in the tree: K3 is NoPE, so its
    // table is the IDENTITY — cos = 1, sin = 0. Baking writes
    // `ctx * qk_rope * 2B * 2` bytes of ones and zeros into the blob, which at
    // ctx 131072 is 33.5 MiB of constants, and it is the ONLY thing in a K3
    // blob that grows with context.
    if !rope_gen {
        m.bake_gen();
    }
    let layers = k3_emit_layers(&c);
    // COVERAGE GATE — and the reason it is here is a bug it would have caught on day one.
    //
    // `checkpoint::validate_coverage` is bidirectional and fatal, and its own header names
    // Kimi-K3 as the case it was written for. It was reachable only from the dense path
    // (`lib.rs`); THIS function went straight from `apply_verify_gate` to `fs::write`. So when
    // the model-level `_apply_output_attn_res` was never emitted, its two weights
    // (`output_attn_res_{norm,proj}`) sat in the checkpoint claimed by nothing, every per-layer
    // golden test stayed green, all 8 ranks agreed, and the model decoded one constant token
    // forever. A missing OP is invisible; the weight it fails to read is not.
    //
    // The gate is run on the DECLARED names before the blob is written, so a program that would
    // drop a weight never reaches disk to be benchmarked by someone else.
    //
    // Truncation is passed through as `block`: under `K3_NLAYERS` the other layers' weights are
    // legitimately uncovered, and without this every truncated emit would read as "an
    // architecture this emitter does not implement".
    let truncated = (layers.len() as u32) < c.layers;
    let covered_layers = truncated.then(|| {
        let first = *layers.first().expect("K3 emits at least one layer") as usize;
        let end = *layers.last().expect("K3 emits at least one layer") as usize + 1;
        assert_eq!(
            layers.len(),
            end - first,
            "K3 layer selection must be contiguous"
        );
        first..end
    });
    match crate::checkpoint::validate_coverage(
        dir,
        K3_PREFIX,
        &m.tensors.iter().map(|t| t.name.clone()).collect::<Vec<_>>(),
        covered_layers,
        K3_INDIRECT,
        K3_PAIRED,
        K3_SYNTHESIZED,
    ) {
        Ok(()) => {}
        Err(e) if emit_config::active().skip_coverage => {
            eprintln!("*** PLOW_SKIP_COVERAGE=1 — EMITTING A MODEL KNOWN TO BE WRONG ***\n{e}");
        }
        Err(e) => {
            eprintln!("kimi_k3: {e}");
            std::process::exit(1);
        }
    }
    // Gate BEFORE the bytes land: a rejected program must never exist on disk.
    let lean = crate::apply_verify_gate(&m, verify);
    std::fs::write(out, m.to_blob()).expect("write k3 devblob");
    eprintln!(
        "kimi_k3: emitted {} layers ({} KDA, {} MLA), tp={tp}, {} tensors, {} decode \
         instructions -> {out}\n  prefill buckets {pf:?} ({} programs incl. decode), ctx={ctx}",
        layers.len(),
        layers
            .iter()
            .filter(|&&l| matches!(c.attn[l as usize], K3Attn::Kda))
            .count(),
        layers
            .iter()
            .filter(|&&l| !matches!(c.attn[l as usize], K3Attn::Kda))
            .count(),
        m.tensors.len(),
        m.progs.last().map(|p| p.insts.len()).unwrap_or(0),
        m.progs.len(),
    );
    write_mla_manifest(&m, out, target, MoeEnc::Mxfp4, &lean);
}

/// The 0-based layer span a K3 emit covers. `K3_NLAYERS` truncates it, and BOTH program kinds are
/// built from this one list, so a truncation cannot leave prefill and decode at different depths.
/// Kimi-K3's coverage waivers — the only checkpoint tensors a correct K3 blob leaves undeclared.
///
/// Each names a mechanism that DOES read the bytes; see `validate_coverage`'s `indirect`
/// contract. Adding an entry here is how a missing op gets hidden, so an addition needs the
/// mechanism, not a plausible story about the weight being unused.
/// Kimi-K3's checkpoint prefix. K3 nests its text tower under a multimodal wrapper, so of the
/// checkpoint's 497,052 language-tower tensors ZERO start with `model.`.
///
/// Shared by the emitter and the coverage gate deliberately: `validate_coverage` filters BOTH
/// sides by the prefix, so a prefix that matches nothing compares two empty sets and passes. A
/// gate keyed on a second copy of this string would silently stop gating the moment the two
/// drifted — which is the same class of failure the gate exists to catch.
pub(crate) const K3_PREFIX: &str = "language_model.model.";

/// Kimi-K3's coverage waivers — the only checkpoint tensors a correct K3 blob leaves undeclared.
/// Names a K3 blob declares that the CHECKPOINT does not ship, because they are produced before
/// the bind. The mirror of [`K3_INDIRECT`]; same rule — each entry names a producer.
const K3_SYNTHESIZED: &[&str] = &[
    // `fold_res_score` (plowrt exec/amd.rs:1912) computes this [H] f32 at load from the
    // checkpoint's `_res_norm`/`_res_proj` pair. It is the twin of the `_res_{norm,proj}`
    // waivers below: one mechanism, one weight consumed, one weight produced.
    "_res_score.weight",
    // Supplied by scripts/kimi_k3_prep.py's `--derived` sidecar, which `shard_files` above
    // deliberately cannot see: it accepts only `model.safetensors` and `model-{i}-of-{n}`, and
    // the sidecar is named `model-idx-derived-*.safetensors` precisely so the COMPILER ignores it
    // while the RUNTIME (which globs every `*.safetensors`) picks it up. So devgen cannot check
    // these here even though they are present on disk at serve time.
    "derived.",
];

/// Conditional waivers — covered only if the consumer is emitted. See `validate_coverage`'s
/// `paired` contract for why these are NOT flat entries in [`K3_INDIRECT`].
///
/// `fold_res_score` turns each `{stem}_res_norm.weight` + `{stem}_res_proj.weight` pair into one
/// `{stem}_res_score.weight`. Three stems exist: `self_attention`, `mlp`, and the model-level
/// `output_attn` — and it was the third whose op went missing. Keying on the produced name means
/// dropping that op un-covers its two weights and fails the emit, which is the whole point.
const K3_PAIRED: &[(&str, &str)] = &[
    ("_res_norm.weight", "_res_score.weight"),
    ("_res_proj.weight", "_res_score.weight"),
];

const K3_INDIRECT: &[&str] = &[
    ".experts.",        // bind_packed_experts, by name pattern (494,592 tensors)
    "q_b_proj.weight",  // absorbed host-side into derived.{q_absorb,q_rope}
    "kv_b_proj.weight", // absorbed host-side into derived.{q_absorb,v_absorb}
];

fn k3_emit_layers(c: &K3Cfg) -> Vec<u32> {
    let (_full, cap, single) = emit_config::active().k3_layer_cfg();
    if let Some(layer) = single {
        assert!(
            layer < c.layers,
            "--k3-layers single:{layer} is outside the model's {} layers",
            c.layers
        );
        return vec![layer];
    }
    let nl = cap.unwrap_or(c.layers).min(c.layers);
    (0..nl).collect()
}

/// [`k3_emit_full`]'s model, with the prefill ladder as a PARAMETER rather than an environment
/// read, and without the file I/O.
///
/// The seam is a parameter for the reason `emit_kda_mixer_ex`'s is: a test that flips an env var
/// races every other test in the binary. It also lets the decode-identity gate build the SAME
/// model twice, once with an empty ladder and once with a full one, and compare the decode
/// program byte for byte — which is the only way to state "the ladder did not move decode" as a
/// fact rather than a hope.
/// BODY ABLATION, and it is a MEASUREMENT INSTRUMENT that produces WRONG TOKENS.
///
/// `PLOW_K3_ABLATE=<opcode>[,<opcode>...]` rewrites the named ops to `Nop` **after** the graph is
/// built, so `stream`, `waits`, `succs`, the counter count and every packet's dispatch width are
/// byte-for-byte what they were — the ONLY thing that goes away is the op's body. Subtracting the
/// ablated run from the full one is therefore that op family's BODY time, the way
/// `PLOW_CHAIN_BYPASS` isolates its CHAIN DEPTH. The two answer different questions and this tree
/// had only the second one on AMD: `PLOW_NV_ABLATE_LO/HI` is NVIDIA-only
/// (`scripts/tune_decode_sweep.sh:399`), which is why K3's per-layer cost had never been attributed.
///
/// Consumers read stale buffers, so tokens are garbage. That is intended and is the same standing
/// as `PLOW_CHAIN_BYPASS`: wrong numerics are a valid instrument for scheduling and for cost.
fn k3_ablate_bodies(m: &mut Model) {
    let Some(ref spec) = emit_config::active().k3_ablate else {
        return;
    };
    let want: Vec<u16> = spec
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    if want.is_empty() {
        return;
    }
    let mut hit = 0usize;
    for p in m.progs.iter_mut() {
        for i in p.insts.iter_mut() {
            if want.contains(&i.op) {
                i.op = packet::dev::DevOp::Nop as u16;
                hit += 1;
            }
        }
    }
    eprintln!("  PLOW_K3_ABLATE: {hit} instruction(s) rewritten to Nop — TOKENS ARE GARBAGE, this is a cost instrument");
}

/// FlashMLA's decode `nsplit` for K3, and the reason it is not simply `glm_nsplit`.
///
/// The work-item count is `(nh_l / gf) * nsplit`, which at TP8 is `3 * nsplit` — so `nsplit = 4`
/// dispatches **12** items and leaves 244 of 256 workgroups empty on all 24 MLA layers. Splitting
/// the KV range further is the only way to fill the machine on this op.
///
/// It is NOT a free widening: `MlaMergeFold` reduces over the `nsplit` partials, so the merge grows
/// as the flash shrinks and the net is a U-shape whose minimum `mla.rs`'s own `NS_CEIL_MEASURED`
/// note records as UNSWEPT at TP8. `PLOW_K3_NS` is therefore the sweep handle and the default is
/// the measured winner; do not change the default without re-running the sweep.
fn k3_nsplit_fallback(ctx: u32) -> u32 {
    if let Some(v) = emit_config::active().k3_ns {
        return v.max(1);
    }
    let _ = ctx;
    // SWEPT on the real gfx942 TP8 asset with FP8 KV and the same interpreter object. At 128K,
    // ns16/ns32/ns64/ns128 measured 81.400/67.417/60.569/60.683 ms TPOT. ns128 brackets the
    // merge-cost reversal. ns64 is tied with ns16 at 149 tokens, wins from 4K through 128K, and
    // retains the established 197/200 GSM8K gate. Keep the explicit override for future GPUs.
    64
}

fn k3_build_model(
    dir: &Path,
    ctx: u32,
    n_cu: u32,
    tp: u32,
    pf: &[u32],
    l2_layout: Option<packet::devbuild::L2Layout>,
) -> Model {
    use crate::k3::{K3MlaCfg, K3ModelCfg, K3MoeCfg};
    let c = cfg_kimi_k3(dir);
    let layers = k3_emit_layers(&c);

    let mut mcfg = K3ModelCfg {
        block: crate::k3::K3BlockCfg {
            hidden: c.hidden,
            eps: c.eps,
            attn_res_block_size: c.attn_res_block,
            situ_beta: c.situ_beta as f32,
            situ_linear_beta: c.situ_linear_beta as f32,
        },
        kda: crate::kda::KdaCfg {
            hidden: c.hidden,
            heads: c.kda_heads,
            head_dim: c.kda_head_dim,
            conv_w: c.kda_conv,
            gate_lower_bound: Some(c.kda_gate_lower_bound as f32),
            eps: c.eps,
            // BV must shrink with the local head count or the state step strands
            // the chip: at tp8 (12 heads) a fixed 16 gives 96 of 256 items.
            bv: if tp >= 8 { 8 } else { 16 },
        },
        mla: K3MlaCfg {
            hidden: c.hidden,
            heads: c.heads,
            q_lora: c.q_lora,
            kv_lora: c.kv_lora,
            qk_rope: c.qk_rope,
            v_head: c.v_head,
            eps: c.eps,
            scale: 1.0 / ((c.qk_nope + c.qk_rope) as f32).sqrt(),
            n_split: k3_nsplit_fallback(ctx),
            gf: 4,
            fp8_kv: emit_config::active().fp8_kv,
        },
        moe: K3MoeCfg {
            hidden: c.hidden,
            latent: c.moe_latent,
            moe_inter: c.moe_inter,
            shared_inter: c.shared_exp * c.moe_inter,
            n_exp: c.n_exp,
            top_k: c.top_k,
            route_flags: u32::from(c.router_sigmoid) | (u32::from(c.renormalize) << 1),
            route_scale: c.route_scale,
            n_group: c.n_group,
            topk_group: c.topk_group,
            enc: MoeEnc::Mxfp4 as u32,
            // The grouped ops passed the full TP8 K3 gate at 4K+16: 103.161 -> 62.893 ms/token,
            // with all 17 dumped logit vectors byte-identical. Keep `0` as the reproducible
            // baseline arm; every other spelling, including unset, ships the measured winner.
            group_decode: true, // Hardcoded ON (was K3_MOE_GROUP, never disabled)
        },
        vocab: c.vocab,
        first_k_dense: c.first_k_dense,
        dense_inter: c.dense_inter,
        prefix: K3_PREFIX.into(),
        tp,
    };

    // THE PROGRAM SET: one per prefill rung, then decode. `k3_emit_full` used to set
    // `prog_t: vec![1]` — decode only — which means a prompt longer than one token has NOTHING to
    // run and the runtime walks it through the decode program a token at a time
    // (`AmdServe::prefill`'s `decode_only` arm, one dispatch per prompt token). That is the whole
    // of TTFT, and with the host phase now measured at 3% of a decode token it is the largest
    // remaining serving gap on this path.
    //
    // DECODE IS BUILT FIRST, and the order is load-bearing rather than tidy. Every program in a
    // blob shares ONE tensor table; `Builder::set_tensor_dedup` lets a later builder adopt the
    // previous table and get the SAME handle back for a name it re-declares, growing the byte
    // count to the max. Building decode into an empty table means its handles are exactly what a
    // decode-only emit produced, so every instruction of the decode program is byte-identical and
    // the buckets can only APPEND (`k3_decode_program_is_unchanged_by_the_prefill_ladder` pins
    // it). The `progs` vector is reordered to buckets-then-decode below, because that is the
    // convention `Model::prog_t` and `manifest` read.
    let mut tensors: Vec<packet::devbuild::TensorDecl> = Vec::new();
    let mut gen = Vec::new();
    let mut built: Vec<packet::devbuild::Program> = Vec::new();
    // BATCHED DECODE. `PLOW_DECODE_BATCH=B` makes the DECODE program carry B INDEPENDENT
    // SEQUENCES rather than one, which is a different thing from a prefill bucket's `t` rows and
    // is why it is paired with `RowKind::Sequences` rather than just a larger `t`
    // (perf-data/archive/k3/k3-batched-decode-design.md §1). B=1 is byte-identical to the pre-batch blob.
    //
    // Above 16 rows the gfx942 GEMV object must carry PLOW_GEMV_WALK: its largest compiled row
    // bucket is 16, and the walk is what covers the remaining rows instead of leaving stale
    // logits. XArgmaxFin carries up to 128 rows across eight peer-data lines.
    // A ladder carries one independent-sequence decode program per rung, including B1. Build the
    // widest first to preserve the old single-rung tensor handles byte for byte. Extents no longer
    // rely on that ordering: scratch rows and independently carried sequence slots are explicit.
    let ladder_on = emit_config::active().decode_ladder.is_some();
    let rungs: Vec<u32> = emit_config::active()
        .decode_rungs()
        .into_iter()
        .map(|r| checked_k3_decode_batch(r, emit_config::active().gemv_walk))
        .collect();
    let dbatch = *rungs.last().expect("decode_rungs is non-empty");
    // Prefill contributes transient rows, never sequence slots. The widest decode rung owns the
    // persistent KDA state, MLA caches, kvlen entries, and sampled-output rows.
    let scratch_rows = pf.iter().copied().max().unwrap_or(1).max(dbatch);
    let sequence_slots = dbatch;
    let mut decode_build_order = Vec::with_capacity(rungs.len());
    decode_build_order.push(dbatch);
    decode_build_order.extend_from_slice(&rungs[..rungs.len() - 1]);

    let mut decode = Vec::with_capacity(rungs.len());
    let mut prefill = Vec::with_capacity(pf.len());
    for (i, &t) in decode_build_order.iter().enumerate() {
        let fallback_ns = k3_nsplit_fallback(ctx);
        let local_heads = c.heads / tp.max(1);
        let shape = format!("mla/dk{}/dr{}/h{}/gf4", c.kv_lora, c.qk_rope, local_heads);
        mcfg.mla.gf = 4;
        mcfg.mla.n_split = if emit_config::active().k3_ns.is_some() {
            fallback_ns
        } else {
            crate::select_amd_attention(n_cu, t, ctx, shape, fallback_ns, fallback_ns).nsplit
        };
        let mut b = Builder::new(n_cu);
        b.set_tensor_dedup(true);
        // PLOW_L2_PLACE: `None` => byte-identical. Until this line the flag reached the dense-GQA
        // builders only, and `kimi_k3` is absent from the arch list that warns about being
        // ignored (`lib.rs:4327`), so setting it on K3 was a silent no-op.
        b.set_l2_placement(l2_layout);
        if crate::emit_is_amd() {
            b.deny_uniseg();
        }
        b.adopt_tensors(tensors.clone());
        crate::k3::emit_k3_model(
            &mut b,
            &mcfg,
            &|l| matches!(c.attn[l as usize], K3Attn::Kda),
            &layers,
            ctx,
            t,
            scratch_rows,
            sequence_slots,
            n_cu,
            // Every rung of a ladder carries independent sequences, including B1. Without a
            // ladder, B1 remains byte-identical unless PLOW_K3_SEQ_ROWS forces the carrier.
            // PLOW_K3_SEQ_ROWS is a
            // BISECTION INSTRUMENT, not a serving knob: at one row every carrier is a no-op by
            // construction (the only slot is slot 0, at offset 0 under either addressing), so a
            // B=1 emit with it on MUST reproduce the known-good B=1 stream token for token. If it
            // does not, the carrier that broke it is separable from batching itself — which is the
            // one question a B>1 run cannot answer, because at B>1 there is no reference stream to
            // compare against.
            if ladder_on || t > 1 || emit_config::active().k3_seq_rows {
                crate::k3::RowKind::Sequences
            } else {
                crate::k3::RowKind::Tokens
            },
        );
        // Every builder re-declares the same NoPE recipes and, under dedup, gets the same handles,
        // so any one of the lists is the whole set. Take the first — decode's — because it is the
        // one a decode-only emit would also have produced.
        if i == 0 {
            gen = b.gen_tensors();
        }
        let prog = b.finish();
        tensors = prog.tensors.clone();
        decode.push((t, prog));
    }
    for &t in pf {
        let mut b = Builder::new(n_cu);
        b.set_tensor_dedup(true);
        b.set_l2_placement(l2_layout);
        b.set_packed_prefill_segments(
            std::env::var("PLOW_SEG_PACKED_PREFILL").ok().as_deref() == Some("1"),
        );
        if crate::emit_is_amd() {
            b.deny_uniseg();
        }
        b.adopt_tensors(tensors.clone());
        crate::k3::emit_k3_model(
            &mut b,
            &mcfg,
            &|l| matches!(c.attn[l as usize], K3Attn::Kda),
            &layers,
            ctx,
            t,
            scratch_rows,
            sequence_slots,
            n_cu,
            crate::k3::RowKind::Tokens,
        );
        let prog = b.finish();
        tensors = prog.tensors.clone();
        prefill.push(prog);
    }
    decode.sort_unstable_by_key(|(t, _)| *t);
    built.extend(prefill);
    built.extend(decode.into_iter().map(|(_, p)| p));
    // Prefill buckets first, then trailing ascending decode rungs. `decode_rung_lo` and the
    // runtime both derive the split from this ordering.
    let prog_t: Vec<u32> = pf.iter().copied().chain(rungs.iter().copied()).collect();

    Model {
        n_cu,
        target: 0,
        tensors,
        progs: built,
        kv_row_insts: Vec::new(),
        prog_t,
        gen,
    }
}

fn checked_k3_decode_batch(decode_batch: u32, gemv_walk: bool) -> u32 {
    let dbatch = decode_batch.max(1);
    assert!(
        dbatch <= packet::devbuild::XARGMAX_MAX_BATCH,
        "K3 PLOW_DECODE_BATCH={dbatch} exceeds the {}-sequence XArgmaxFin ceiling",
        packet::devbuild::XARGMAX_MAX_BATCH
    );
    assert!(
        dbatch <= 16 || gemv_walk,
        "K3 PLOW_DECODE_BATCH={dbatch} requires PLOW_GEMV_WALK=1 above 16 rows"
    );
    dbatch
}

/// The prefill rungs a K3 emit builds programs for.
///
/// `T` is a COMPILE-TIME constant of a packet, so the ladder is the only way a 20-token prompt and
/// a 4096-token one can both avoid paying for the other's program. The rungs are GLM's — this
/// family's prefill object is gfx950-only and there is no K3 measurement to place treads with, so
/// re-deriving them here would be guessing.
///
/// ON BY DEFAULT, unlike `PLOW_MLA_PREFILL`. That knob is off because the GLM MLA prefill arm can
/// be built at an ATTENTION-ONLY scope that never writes `act.logits` — a blob whose prefill
/// programs cannot sample while `Engine::has_prefill()` is true. K3 has no such scope: every
/// bucket here is a whole model, embed through argmax, so there is no half-built state to opt into.
///
///   * unset / `1` / `full` — the whole ladder, capped at `ctx`;
///   * `0`                  — decode only, byte-identical to before this path existed;
///   * `512,1024`           — those rungs only.
///
/// The list form is not cosmetic: every activation is declared for the WIDEST bucket, and
/// `act.pf.moe.part` alone is `T * top_k * latent` f32 — **1.9 GiB at T = 8192** on the shipped
/// geometry. A deployment that will only ever see 1k prompts should not pay for the 8192 rung.
pub(crate) fn k3_prefill_buckets(ctx: u32) -> Vec<u32> {
    match emit_config::active().k3_prefill.as_deref() {
        Some("0") => Vec::new(),
        None | Some("") | Some("1") | Some("full") => glm_prefill_buckets(ctx),
        Some(list) => list
            .split(',')
            .filter_map(|s| s.trim().parse::<u32>().ok())
            .filter(|&x| x > 1 && x <= ctx)
            .collect(),
    }
}

pub(crate) fn kimi_k3_emit(dir: &Path, ctx: u32, tp: u32, block_spec: Option<&str>) -> ! {
    let c = cfg_kimi_k3(dir);
    let (hdrs, have, total) = k3_shard_headers(dir);
    let mismatches = k3_config_vs_tensors(&c, &hdrs);

    eprintln!("kimi_k3: config ACCEPTED, emission REFUSED.\n");
    eprintln!("  checkpoint  {}", dir.display());
    if total == 0 {
        eprintln!("  shards      none on disk — every dimension below is CONFIG-ONLY, unverified");
    } else {
        eprintln!(
            "  shards      {have}/{total} readable, {} tensors{}",
            hdrs.len(),
            if have < total {
                "  (PARTIAL: a tensor's absence proves nothing)"
            } else {
                ""
            }
        );
    }
    eprintln!(
        "  text tower  {} layers = {} MLA + {} KDA | hidden {} | heads {} | vocab {} | ctx {ctx} | tp {tp}",
        c.layers, c.n_mla(), c.n_kda(), c.hidden, c.heads, c.vocab
    );
    eprintln!(
        "  MLA         q_lora {} kv_lora {} qk {}+{} v {} | nope={} out_gate={} rope_theta={}",
        c.q_lora,
        c.kv_lora,
        c.qk_nope,
        c.qk_rope,
        c.v_head,
        c.mla_nope,
        c.mla_out_gate,
        match c.rope_theta {
            Some(t) => format!("{t}"),
            None => "ABSENT".into(),
        }
    );
    eprintln!(
        "  KDA         {} heads x {} dim, conv k={}, {} gate, lower bound {}",
        c.kda_heads,
        c.kda_head_dim,
        c.kda_conv,
        if c.kda_full_rank_gate {
            "full-rank"
        } else {
            "low-rank"
        },
        c.kda_gate_lower_bound
    );
    eprintln!(
        "  MoE         {} routed (top-{}) + {} shared | inter {} | LATENT {} | norm={} | dense L<{} inter {}",
        c.n_exp, c.top_k, c.shared_exp, c.moe_inter, c.moe_latent, c.latent_norm,
        c.first_k_dense, c.dense_inter
    );
    eprintln!(
        "  router      {} | renorm={} | scale {} | groups {}/{} | act {:?}",
        if c.router_sigmoid {
            "sigmoid"
        } else {
            "softmax"
        },
        c.renormalize,
        c.route_scale,
        c.topk_group,
        c.n_group,
        c.hidden_act
    );
    eprintln!(
        "  quant       {} | {} bits | group {} | routed experts ONLY (attn/shared/dense/lm_head stay bf16)",
        c.quant_format, c.quant_bits, c.quant_group
    );
    eprintln!(
        "  attn map    0-based MLA layers {:?}{}",
        c.attn
            .iter()
            .enumerate()
            .filter(|(_, &k)| k == K3Attn::Mla)
            .map(|(i, _)| i)
            .take(6)
            .collect::<Vec<_>>(),
        if c.n_mla() > 6 { " …" } else { "" }
    );
    if let Some(spec) = block_spec {
        eprintln!("  --block     {spec:?} (accepted, but no layer kind is emittable — see below)");
    }
    if let Some(vis) = &c.vision {
        eprintln!(
            "\nSCOPE REFUSAL: this checkpoint is MULTIMODAL and plow implements the TEXT tower \
             only.\n  refused     MoonViT vision_tower ({} layers x {} hidden) + mm_projector \
             ({:?}).\n              Not skipped, not partially compiled: a text-only blob for a \
             multimodal\n              checkpoint loads, runs, and is wrong on every image prompt. \
             Strip\n              `vision_config` from config.json to ask for the text tower \
             explicitly.",
            vis.layers, vis.hidden, vis.projector
        );
    }

    if mismatches.is_empty() {
        eprintln!(
            "\n  config vs tensors: AGREE on every dimension the {} readable tensors can speak to.",
            hdrs.len()
        );
    } else {
        eprintln!(
            "\n  config vs tensors: {} DISAGREEMENT(S). TRUST THE TENSORS (GLM-5.2 lesson):",
            mismatches.len()
        );
        for m in &mismatches {
            eprintln!("    - {m}");
        }
    }

    // The other half of an honest refusal: what is ALREADY there. A gap list on its own invites
    // the next agent to rebuild machinery that exists — the mirror image of §4's "an arm exists
    // and nothing routes to it".
    eprintln!("\nALREADY COVERED — do not rebuild these:");
    eprintln!(
        "  mxfp4 routed experts  the MoE expert path already carries a weight ENCODING field \
         (MoeEnc::Mxfp4 = 2,\n                        i[6] decode / i[3] prefill) with an emitter \
         selector, and `wave_dot_mxfp4`\n                        (runtime/amd/op_moe.h:395) is \
         w4a16 — bf16 activations against packed fp4,\n                        exactly this \
         checkpoint's scheme (`input_activations: null`). The on-disk\n                        \
         layout is byte-exact: weight_packed [N, K/2], weight_scale [N, K/{}] u8.\n                 \
        Nothing to pack, nothing to convert.",
        c.quant_group.max(1)
    );
    eprintln!(
        "  E8M0 bias             127, per knob-contract §2, and CONFIRMED empirically from this \
         checkpoint:\n                        w1 scale bytes span 115-122, i.e. 2^-12..2^-5. Under \
         a bias of 0 the same\n                        bytes would mean 2^115, so the convention is \
         not in doubt."
    );
    eprintln!(
        "  router width          {} experts is inside the analysed bound: the LDS note at \
         runtime/amd/op_moe.h:50-56\n                        works the arena out to n_exp ~12000 \
         and the packed key gives the id 20 bits.\n                        (The \"n_exp<=256 is \
         tiny\" remark at op_moe.h:93 is stale, not a limit.)",
        c.n_exp
    );
    eprintln!(
        "  MLA geometry          q_lora/kv_lora/qk_nope/qk_rope/v_head are the SAME schema \
         cfg_glm parses, and\n                        every one agrees with the tensors. The \
         absorbed form (q_absorb/v_absorb) is\n                        already produced and \
         numerically verified by scripts/kimi_k3_prep.py."
    );

    let gaps = k3_gaps(&c);
    let (closed, open): (Vec<_>, Vec<_>) = gaps.iter().partition(|g| g.done.is_some());

    // CLOSED FIRST, and in full. The point of this section is to stop the next reader building
    // what already runs — which is a failure this report has actually caused, not a hypothetical.
    if !closed.is_empty() {
        eprintln!(
            "\nCLOSED — {} capabilities that WERE on this list and now LAND, each with the \
             real-weight\nevidence. DO NOT REBUILD THESE. Read the `done:` line before writing any \
             opcode.\n",
            closed.len()
        );
        for (i, g) in closed.iter().enumerate() {
            eprintln!("C{:<2} {}  [{}]", i + 1, g.what, g.scope);
            for line in textwrap72(g.done.unwrap()) {
                eprintln!("      {line}");
            }
            eprintln!("      residual work, if any:");
            for line in textwrap72(&g.fix.split_whitespace().collect::<Vec<_>>().join(" ")) {
                eprintln!("        {line}");
            }
            eprintln!();
        }
    }

    eprintln!(
        "\nMISSING CAPABILITIES — {} of them, ranked (blocker first). Each names where the fix \
         goes.\n",
        open.len()
    );
    for (i, g) in open.iter().enumerate() {
        eprintln!("{:>2}. {}  [{}]", i + 1, g.what, g.scope);
        for line in textwrap72(&g.why) {
            eprintln!("      {line}");
        }
        eprintln!("      fix:");
        for line in textwrap72(&g.fix.split_whitespace().collect::<Vec<_>>().join(" ")) {
            eprintln!("        {line}");
        }
        eprintln!();
    }
    panic!(
        "kimi_k3: {} unimplemented capabilities (listed above; {} further capabilities are CLOSED \
         and must not be rebuilt); no correct devblob exists for this checkpoint{}.",
        open.len(),
        closed.len(),
        if c.vision.is_some() {
            ", and its vision tower is out of scope and REFUSED (text-only)"
        } else {
            ""
        }
    );
}

/// Minimal greedy wrap so the capability report stays readable in a terminal.
fn textwrap72(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut line = String::new();
    for w in s.split_whitespace() {
        if !line.is_empty() && line.len() + 1 + w.len() > 92 {
            out.push(std::mem::take(&mut line));
        }
        if !line.is_empty() {
            line.push(' ');
        }
        line.push_str(w);
    }
    if !line.is_empty() {
        out.push(line);
    }
    out
}

/// Kimi-K3 FRONT-END tests. There is no emit to test; what these lock down is the config
/// contract, because every one of them is a place a wrong answer would be silent.
#[cfg(test)]
mod kimi_k3_tests {
    use super::*;

    #[test]
    fn decode_batch_above_sixteen_requires_walk_and_caps_at_xargmax_limit() {
        assert_eq!(checked_k3_decode_batch(0, false), 1);
        assert_eq!(checked_k3_decode_batch(16, false), 16);
        assert_eq!(checked_k3_decode_batch(32, true), 32);
        assert_eq!(checked_k3_decode_batch(128, true), 128);
        assert!(std::panic::catch_unwind(|| checked_k3_decode_batch(17, false)).is_err());
        assert!(std::panic::catch_unwind(|| checked_k3_decode_batch(129, true)).is_err());
    }

    /// A faithful miniature of the real `config.json`: same key spellings, same nesting, same
    /// 1-based layer lists, scaled to 6 layers (2 MLA at 1-based {3,6}, 4 KDA).
    fn k3_json(patch: &[(&str, &str)]) -> Value {
        let base = r#"{
          "model_type": "kimi_k3",
          "architectures": ["KimiK3ForConditionalGeneration"],
          "vision_config": {"vt_num_hidden_layers": 27, "vt_hidden_size": 1024,
                            "mm_projector_type": "patchmergerv2"},
          "text_config": {
            "model_type": "kimi_linear",
            "hidden_size": 256, "num_attention_heads": 8, "num_hidden_layers": 6,
            "vocab_size": 1000, "intermediate_size": 512, "rms_norm_eps": 1e-5,
            "q_lora_rank": 64, "kv_lora_rank": 32, "qk_nope_head_dim": 16,
            "qk_rope_head_dim": 8, "v_head_dim": 16,
            "mla_use_nope": true, "mla_use_output_gate": true,
            "num_experts": 32, "num_experts_per_token": 4, "num_shared_experts": 2,
            "moe_intermediate_size": 96, "routed_expert_hidden_size": 128,
            "latent_moe_use_norm": true, "first_k_dense_replace": 1,
            "routed_scaling_factor": 1.0, "moe_router_activation_func": "sigmoid",
            "moe_renormalize": true, "num_expert_group": 1, "topk_group": 1,
            "hidden_act": "situ", "activation_situ_beta": 4.0,
            "activation_situ_linear_beta": 25.0, "attn_res_block_size": 12,
            "linear_attn_config": {
              "num_heads": 8, "head_dim": 32, "short_conv_kernel_size": 4,
              "use_full_rank_gate": true, "gate_lower_bound": -5.0,
              "full_attn_layers": [3, 6], "kda_layers": [1, 2, 4, 5]
            },
            "quantization_config": {"format": "mxfp4-pack-quantized",
              "config_groups": {"group_0": {"weights": {"group_size": 32, "num_bits": 4}}}}
          }
        }"#;
        let mut v: Value = serde_json::from_str(base).unwrap();
        for (path, val) in patch {
            let seg: Vec<&str> = path.split('/').collect();
            let mut cur = &mut v;
            for s in &seg[..seg.len() - 1] {
                cur = cur.get_mut(s).unwrap();
            }
            let last = seg[seg.len() - 1];
            match *val {
                "<remove>" => {
                    cur.as_object_mut().unwrap().remove(last);
                }
                s => {
                    cur[last] = serde_json::from_str(s).unwrap();
                }
            }
        }
        v
    }

    /// The layer lists are 1-BASED (`configuration_kimi_k3.py::is_kda_layer` tests
    /// `(layer_idx + 1) in kda_layers`) and the real checkpoint proves it: MLA tensors live on
    /// 0-based layers 3, 7, 11, … while `full_attn_layers` starts at 4. An off-by-one here binds
    /// `q_a_proj` to a layer that ships `q_proj` — or, worse, does not fail and mixes the two.
    #[test]
    fn layer_lists_are_one_based_and_partition_the_tower() {
        let c = k3_cfg_from(&k3_json(&[]));
        assert_eq!(
            c.attn,
            vec![
                K3Attn::Kda, // 1-based 1
                K3Attn::Kda, // 2
                K3Attn::Mla, // 3
                K3Attn::Kda, // 4
                K3Attn::Kda, // 5
                K3Attn::Mla, // 6
            ]
        );
        assert_eq!((c.n_mla(), c.n_kda()), (2, 4));
        assert_eq!(c.attn.len(), c.layers as usize);
    }

    /// A checkpoint DIRECTORY holding the miniature config, for the two tests that drive
    /// `k3_build_model` end to end. There is no K3 checkpoint on this machine — only weights are
    /// missing, and `k3_build_model` never reads any.
    fn k3_dir(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("plow_k3_{name}"));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        // head_dim 64, not the fixture's 32: `emit_kda_mixer` refuses a head_dim that is not a
        // multiple of the 64-lane wave, and these two tests are the only ones here that EMIT.
        let mut patch = vec![("text_config/linear_attn_config/head_dim", "64")];
        if name == "attention_profiles" {
            patch.push(("text_config/num_attention_heads", "24"));
        }
        let cfg = k3_json(&patch);
        std::fs::write(d.join("config.json"), cfg.to_string()).unwrap();
        d
    }

    /// **THE DECODE PROGRAM IS UNCHANGED BY THE PREFILL LADDER — byte for byte.**
    ///
    /// This is the whole reason decode is built FIRST into an empty tensor table. Every program in
    /// a blob shares one table, so a bucket that declared a handle ahead of decode would renumber
    /// every `t[]` slot in the decode program: same graph, different bytes, and a regression no
    /// op-census test would see. Building decode first means the buckets can only APPEND.
    ///
    /// It compares the SERIALIZED instruction stream, not an op count. `DevInst` carries the
    /// tensor handles, the immediates and the counter wiring; two programs whose op sequences agree
    /// can still differ in every one of those.
    #[test]
    fn the_prefill_ladder_leaves_the_decode_program_byte_identical() {
        // This is the test the PLOW_GLM_WGFIT race actually broke: the emitted
        // block counts depend on the narrowing, so it must not run while a
        // sibling holds the knob (observed: "decode inst 77: block count
        // left: 256, right: 32", ~1 full-suite run in 7).
        let _env = crate::test_env::env_guard();
        let d = k3_dir("ladder");
        let bare = k3_build_model(&d, 4096, 256, 1, &[], None);
        let laddered = k3_build_model(&d, 4096, 256, 1, &[128, 512, 1024], None);

        assert_eq!(
            bare.prog_t,
            vec![1],
            "no ladder ⇒ decode only, as before this path existed"
        );
        assert_eq!(
            laddered.prog_t,
            vec![128, 512, 1024, 1],
            "buckets ascending, decode LAST"
        );
        assert_eq!(laddered.progs.len(), 4);

        let a = bare.progs.last().unwrap();
        let b = laddered.progs.last().unwrap();
        assert_eq!(a.insts.len(), b.insts.len(), "decode op count moved");
        for (i, (x, y)) in a.insts.iter().zip(b.insts.iter()).enumerate() {
            assert_eq!(x.op, y.op, "decode inst {i}: opcode");
            assert_eq!(x.t, y.t, "decode inst {i}: tensor handles renumbered");
            assert_eq!(x.i, y.i, "decode inst {i}: immediates");
            assert_eq!(x.f, y.f, "decode inst {i}: floats");
            assert_eq!(x.blocks, y.blocks, "decode inst {i}: block count");
        }
        assert_eq!(a.stream, b.stream, "decode per-CU streams");
        assert_eq!(a.waits, b.waits, "decode counter waits");
        assert_eq!(a.succs, b.succs, "decode counter successors");

        // The shared table is a SUPERSET whose common prefix is decode's own, in order — that is
        // what makes the handles above stable. Sizes GROW (activations are declared for the widest
        // bucket); names and order do not move.
        for (i, td) in bare.tensors.iter().enumerate() {
            assert_eq!(td.name, laddered.tensors[i].name, "tensor {i} moved");
            assert!(
                laddered.tensors[i].bytes >= td.bytes,
                "tensor {} shrank",
                td.name
            );
        }
        assert!(
            laddered.tensors.len() > bare.tensors.len(),
            "buckets declare their own scratch"
        );
        // And the ladder actually widened the shared activations rather than merely appending.
        let ring = |m: &Model| {
            m.tensors
                .iter()
                .find(|t| t.name == "kv.blkres")
                .unwrap()
                .bytes
        };
        assert_eq!(
            ring(&laddered),
            1024 * ring(&bare),
            "the ring is [T][nb_cap][hidden]"
        );
    }

    #[test]
    fn prefill_rows_do_not_expand_sequence_state() {
        let _guard = crate::test_env::env_guard();
        let d = k3_dir("row_extents");
        let base = k3_build_model(&d, 8192, 256, 2, &[], None);
        let bytes = |m: &Model, name: &str| {
            m.tensors
                .iter()
                .find(|t| t.name == name)
                .unwrap_or_else(|| panic!("missing tensor {name}"))
                .bytes
        };

        let wide = {
            let _scope = crate::test_env::EnvScope::set(&[
                ("PLOW_DECODE_BATCH_LADDER", "1,128"),
                ("PLOW_GEMV_WALK", "1"),
            ]);
            k3_build_model(&d, 8192, 256, 2, &[8192], None)
        };
        let c = cfg_kimi_k3(&d);

        assert_eq!(bytes(&wide, "in.kvlen"), 128 * 4);
        assert_eq!(bytes(&wide, "act.x"), 8192 * c.hidden as u64 * 2);
        assert_eq!(
            bytes(&wide, "act.og_tp"),
            8192 * c.hidden as u64 * 2,
            "peer scratch follows the largest row program"
        );
        assert_eq!(
            bytes(&wide, "kv.0.state"),
            128 * bytes(&base, "kv.0.state"),
            "KDA state follows decode sequence slots, not prefill rows"
        );
        assert_eq!(
            bytes(&wide, "kv.2.ckv"),
            128 * bytes(&base, "kv.2.ckv"),
            "MLA cache follows decode sequence slots, not prefill rows"
        );
    }

    #[test]
    fn decode_only_b1_keeps_single_row_and_slot_extents() {
        let _guard = crate::test_env::env_guard();
        let d = k3_dir("decode_only_extents");
        let m = k3_build_model(&d, 8192, 256, 2, &[], None);
        let c = cfg_kimi_k3(&d);
        let bytes = |name: &str| {
            m.tensors
                .iter()
                .find(|t| t.name == name)
                .unwrap_or_else(|| panic!("missing tensor {name}"))
                .bytes
        };

        assert_eq!(m.prog_t, [1]);
        assert_eq!(bytes("in.kvlen"), 4);
        assert_eq!(bytes("act.x"), c.hidden as u64 * 2);
        assert_eq!(bytes("act.og_tp"), c.hidden as u64 * 2);
        assert_eq!(
            bytes("kv.2.ckv"),
            8192 * c.kv_lora as u64 * 2,
            "one sequence owns one context-length MLA cache"
        );
    }

    #[test]
    fn k3_decode_ladder_is_trailing_ascending_and_every_rung_is_sequence_state() {
        let _guard = crate::test_env::env_guard();
        let _scope = crate::test_env::EnvScope::set(&[
            ("PLOW_DECODE_BATCH_LADDER", "1,3,7,16,32,64,128"),
            ("PLOW_GEMV_WALK", "1"),
        ]);
        let d = k3_dir("decode_ladder");
        let m = k3_build_model(&d, 4096, 256, 1, &[128], None);

        assert_eq!(m.prog_t, [128, 1, 3, 7, 16, 32, 64, 128]);
        for (&t, p) in m.prog_t[1..].iter().zip(&m.progs[1..]) {
            let state_steps: Vec<_> = p
                .insts
                .iter()
                .filter(|i| {
                    i.op == DevOp::KdaStateStep as u16 || i.op == DevOp::KdaStateStepG as u16
                })
                .collect();
            assert!(
                !state_steps.is_empty(),
                "decode rung T={t} has no KDA state"
            );
            assert!(
                state_steps.iter().all(|i| i.i[4] & 2 != 0),
                "decode rung T={t} lost independent-sequence KDA state"
            );
        }
        let parked = m
            .tensors
            .iter()
            .find(|t| t.name == "in.parked")
            .expect("ladder must declare the parked mask");
        assert_eq!(parked.bytes, 128 * 4);
    }

    fn publish_attention_test_record(root: &std::path::Path, rung: u32, nsplit: u32, shape: &str) {
        let record = tunedb::AttentionMeasurement {
            cell: tunedb::AttentionCell {
                hardware: "amd/gfx950/mi350x".into(),
                n_cu: 256,
                decode_rung: rung,
                kv_bucket: tunedb::KvBucket::K8,
                shape: shape.into(),
            },
            algorithm: tunedb::AttentionAlgorithm::SplitReduce,
            nsplit,
            digests: tunedb::Digests {
                implementation: "test-unprobed".into(),
                interpreter: "test-unprobed".into(),
                toolchain: "test-unprobed".into(),
                oracle: tunedb::ATTENTION_ORACLE.into(),
            },
            stats: tunedb::Stats::from_samples(vec![10_000.0; 5]).unwrap(),
            correctness: tunedb::Correctness::Pass,
            state: tunedb::RecordState::Provisional,
            campaign: "attention-rung-test".into(),
        };
        tunedb::TuneStore::new(root)
            .publish_attention("amd/gfx950/mi350x", vec![record])
            .unwrap();
    }

    fn assert_decode_mla_policy(model: &Model, rungs: &[(u32, u32)]) {
        assert_eq!(model.prog_t, rungs.iter().map(|x| x.0).collect::<Vec<_>>());
        for (program, &(batch, nsplit)) in model.progs.iter().zip(rungs) {
            let flashes: Vec<_> = program
                .insts
                .iter()
                .filter(|i| i.op == DevOp::FlashMlaDecode as u16)
                .collect();
            let merges: Vec<_> = program
                .insts
                .iter()
                .filter(|i| i.op == DevOp::MlaMergeFold as u16)
                .collect();
            assert!(!flashes.is_empty());
            assert_eq!(flashes.len(), merges.len());
            for (flash, merge) in flashes.into_iter().zip(merges) {
                assert_eq!(flash.i[4], nsplit, "B{batch} flash nsplit");
                assert_eq!(flash.i[7], 4, "B{batch} flash group factor");
                assert_eq!(merge.i[4], nsplit, "B{batch} merge nsplit");
                let groups = (flash.i[1] / flash.i[7].max(1)).max(1);
                let want_blocks = (batch * groups * nsplit).min(256) as u16;
                assert_eq!(flash.blocks, want_blocks, "B{batch} flash blocks");
            }
        }
    }

    #[test]
    fn k3_attention_profiles_select_each_rung_and_size_scratch() {
        let _guard = crate::test_env::env_guard();
        let d = k3_dir("attention_profiles");
        let db = d.join("tuning");
        let shape = "mla/dk32/dr8/h12/gf4";
        publish_attention_test_record(&db, 1, 32, shape);
        publish_attention_test_record(&db, 8, 32, shape);
        let dbs = db.to_string_lossy().into_owned();

        let tuned = {
            let _scope = crate::test_env::EnvScope::set(&[
                ("PLOW_DECODE_BATCH_LADDER", "1,8"),
                ("PLOW_TUNEDB", &dbs),
            ]);
            k3_build_model(&d, 8192, 256, 2, &[], None)
        };
        assert_decode_mla_policy(&tuned, &[(1, 32), (8, 32)]);
        let scratch_bytes = |name: &str| {
            tuned
                .tensors
                .iter()
                .find(|t| t.name == name)
                .unwrap_or_else(|| panic!("missing MLA partial scratch {name}"))
                .bytes
        };
        assert_eq!(scratch_bytes("act.l2.o_part"), 12 * 32 * 32 * 4);
        assert_eq!(
            scratch_bytes("act.pf.o_part"),
            8 * 12 * 32 * 32 * 4,
            "widest selected decode scratch extent is retained"
        );

        let fallback = {
            let _scope = crate::test_env::EnvScope::set(&[
                ("PLOW_DECODE_BATCH_LADDER", "1,8"),
                ("PLOW_TUNEDB", ""),
            ]);
            k3_build_model(&d, 8192, 256, 2, &[], None)
        };
        assert_decode_mla_policy(&fallback, &[(1, 64), (8, 64)]);

        let pinned = {
            let _scope = crate::test_env::EnvScope::set(&[
                ("PLOW_DECODE_BATCH_LADDER", "1,8"),
                ("PLOW_TUNEDB", &dbs),
                ("PLOW_K3_NS", "7"),
            ]);
            k3_build_model(&d, 8192, 256, 2, &[], None)
        };
        assert_decode_mla_policy(&pinned, &[(1, 7), (8, 7)]);
    }

    #[test]
    fn packed_prefill_segmentation_does_not_split_decode_rungs() {
        let _guard = crate::test_env::env_guard();
        let _scope = crate::test_env::EnvScope::set(&[
            ("PLOW_SEG_PACKED_PREFILL", "1"),
            ("PLOW_DECODE_BATCH_LADDER", "1,4,8"),
            ("PLOW_GEMV_WALK", "1"),
        ]);
        let d = k3_dir("packed_prefill_segments");
        let l2 = packet::devbuild::L2Layout {
            sms: 32,
            domains: 8,
            map: packet::devbuild::L2Map::RoundRobin,
        };
        let m = k3_build_model(&d, 4096, 256, 2, &[128], Some(l2));

        assert_eq!(m.prog_t, [128, 1, 4, 8]);
        assert_eq!(m.progs[0].l2_domains, 0);
        assert!(m.progs[0].gq_seg_ofs.len() > 2);
        for p in &m.progs[1..] {
            assert_eq!(p.l2_domains, 8);
            assert_eq!(p.gq_seg_ofs.len(), 9);
        }
    }

    /// Every program must address the same peer slot B. The host has one peer layout for the
    /// whole blob and binds `act.dg_tp` at the blob-wide offset; a per-program `t*hidden*2`
    /// immediate makes decode reduce unrelated memory whenever a prefill ladder is present.
    #[test]
    fn k3_tp_peer_slot_is_program_invariant() {
        let d = k3_dir("tp_slot");
        let m = k3_build_model(&d, 4096, 256, 2, &[128, 512], None);
        let hidden = cfg_kimi_k3(&d).hidden;
        let want = 512 * hidden * 2;
        for (pi, p) in m.progs.iter().enumerate() {
            let slots: std::collections::BTreeSet<u32> = p
                .insts
                .iter()
                .filter(|i| i.op == DevOp::XReduce as u16 || i.op == DevOp::XReduceTwoShot as u16)
                .map(|i| i.i[2])
                .collect();
            assert_eq!(
                slots,
                [0, want].into_iter().collect(),
                "program {pi} (T={}) disagrees with the blob-wide peer layout",
                m.prog_t[pi]
            );
        }
    }

    /// The ladder itself: rungs are capped at `ctx`, and `K3_PREFILL=0` is the identity.
    ///
    /// The cap is not cosmetic — a program for `T > ctx` can never be invoked, and every
    /// activation in the blob is declared for the WIDEST bucket, so an uncapped ladder charges a
    /// 4k-context deployment for an 8192-row program it cannot run.
    #[test]
    fn the_prefill_ladder_is_capped_at_the_context() {
        assert_eq!(
            k3_prefill_buckets(131072),
            vec![128, 512, 1024, 2048, 4096, 8192]
        );
        assert_eq!(k3_prefill_buckets(2048), vec![128, 512, 1024, 2048]);
        assert_eq!(
            k3_prefill_buckets(64),
            Vec::<u32>::new(),
            "no rung fits — decode only"
        );
    }

    /// Every bucket is a WHOLE MODEL: embed through argmax, with the grouped-MoE FFN.
    ///
    /// GLM's `PLOW_MLA_PREFILL=1` has an attention-only scope that stops at the post-attention
    /// norm and never writes `act.logits` — and `Engine::has_prefill()` is still true, so the
    /// runtime selects those programs and samples from a buffer nothing wrote. There is no such
    /// scope here, and this is what says so.
    #[test]
    fn every_prefill_bucket_is_a_whole_model() {
        let d = k3_dir("whole");
        let m = k3_build_model(&d, 4096, 256, 1, &[128, 512], None);
        let c = cfg_kimi_k3(&d);
        for (pi, p) in m.progs.iter().enumerate() {
            let t = m.prog_t[pi];
            let n = |o: DevOp| p.insts.iter().filter(|i| i.op == o as u16).count();
            assert_eq!(n(DevOp::Embed), 1, "prog {pi} (T={t}): no embed prologue");
            assert_eq!(n(DevOp::ArgmaxFin), 1, "prog {pi} (T={t}): cannot sample");
            // Two AttnRes mixes on every layer, on every program. An AttnRes present in one bucket
            // and not another would make the two phases compute DIFFERENT MODELS.
            // ...and ONE model-level mix (`_apply_output_attn_res`) on every program too — the
            // site whose absence left `model.norm` reading only the post-snapshot partial sum.
            assert_eq!(
                n(DevOp::AttnRes),
                2 * c.layers as usize + 1,
                "prog {pi} (T={t})"
            );
            // The FFN half is present on both, in the spelling that phase has a kernel for.
            let moe_layers = (c.layers - c.first_k_dense) as usize;
            if t == 1 {
                assert_eq!(n(DevOp::MoeRouterTopk), moe_layers);
                assert_eq!(n(DevOp::MoeCombine), moe_layers);
                assert_eq!(n(DevOp::MoeCombinePf), 0);
            } else {
                assert_eq!(n(DevOp::MoeRouterTopkPf), moe_layers);
                assert_eq!(n(DevOp::MoeCombinePf), moe_layers);
                assert_eq!(n(DevOp::MoeCombine), 0);
            }
        }
    }

    /// The map is FIRST-CLASS data, not a count: a config whose MLA layers sit at different
    /// indices must produce a different map even though the counts are identical.
    #[test]
    fn attn_map_is_not_a_count() {
        let a = k3_cfg_from(&k3_json(&[]));
        let b = k3_cfg_from(&k3_json(&[
            ("text_config/linear_attn_config/full_attn_layers", "[1, 4]"),
            ("text_config/linear_attn_config/kda_layers", "[2, 3, 5, 6]"),
        ]));
        assert_eq!((a.n_mla(), a.n_kda()), (b.n_mla(), b.n_kda()));
        assert_ne!(a.attn, b.attn, "same counts must not mean the same model");
    }

    #[test]
    #[should_panic(expected = "appears in BOTH")]
    fn overlapping_layer_lists_are_rejected() {
        k3_cfg_from(&k3_json(&[(
            "text_config/linear_attn_config/kda_layers",
            "[1, 2, 3, 4, 5]",
        )]));
    }

    #[test]
    #[should_panic(expected = "are in neither list")]
    fn incomplete_layer_lists_are_rejected() {
        // Dropping 1-based layer 5 leaves 0-based layer 4 unclassified. Deriving KDA as the
        // complement of full_attn_layers would hide this; the partition check is the point.
        k3_cfg_from(&k3_json(&[(
            "text_config/linear_attn_config/kda_layers",
            "[1, 2, 4]",
        )]));
    }

    #[test]
    #[should_panic(expected = "num_hidden_layers is 6")]
    fn out_of_range_layer_index_is_rejected() {
        k3_cfg_from(&k3_json(&[(
            "text_config/linear_attn_config/full_attn_layers",
            "[3, 7]",
        )]));
    }

    /// `routed_expert_hidden_size` (3584 on the real model), NOT `moe_intermediate_size`, is the
    /// routed-expert GEMM's K. Verified against the checkpoint: `experts.0.w1.weight_packed` is
    /// [moe_inter, routed_expert_hidden_size/2].
    #[test]
    fn routed_experts_run_at_the_latent_width() {
        let c = k3_cfg_from(&k3_json(&[]));
        assert_eq!(c.moe_latent, 128);
        assert_eq!(c.moe_inter, 96);
        assert_ne!(c.moe_latent, c.moe_inter);
        assert_ne!(c.moe_latent, c.hidden);
        // The shape predicate the emitter will have to satisfy, spelled out.
        let (k, n) = (c.moe_latent as i64, c.moe_inter as i64);
        assert_eq!(vec![n, k / 2], vec![96, 64], "w1.weight_packed = [N, K/2]");
        assert_eq!(
            vec![n, k / c.quant_group as i64],
            vec![96, 4],
            "w1.weight_scale = [N, K/group]"
        );
    }

    /// The Kimi spellings differ from DeepSeek's. Reading `n_routed_experts` here would either
    /// hard-error or, with a default, silently compile a dense model.
    #[test]
    #[should_panic(expected = "missing required field \"num_experts\"")]
    fn deepseek_moe_spellings_are_not_accepted() {
        k3_cfg_from(&k3_json(&[("text_config/num_experts", "<remove>")]));
    }

    /// `rope_theta` is ABSENT from this config and must stay `None`. Silently applying GLM's RoPE
    /// to a NoPE model is a silent-corruption bug, not a missing feature; `cfg_glm`'s matching
    /// half is `require_mla_rope` (see `mla_rope_tests` in lib.rs).
    #[test]
    fn absent_rope_theta_is_none_not_a_default() {
        let c = k3_cfg_from(&k3_json(&[]));
        assert_eq!(c.rope_theta, None);
        assert!(c.mla_nope);
        assert!(
            k3_gaps(&c).iter().any(|g| g.what.contains("NO positional")),
            "a NoPE model must produce an explicit gap"
        );
    }

    /// Vision is recorded and REFUSED by name — never silently dropped.
    #[test]
    fn vision_is_recorded_for_explicit_refusal() {
        let c = k3_cfg_from(&k3_json(&[]));
        let v = c
            .vision
            .expect("vision_config must be recorded, not ignored");
        assert_eq!((v.layers, v.hidden), (27, 1024));
        // A text-only re-export has none and must not be flagged.
        let text_only = k3_cfg_from(&k3_json(&[("vision_config", "<remove>")]));
        assert!(text_only.vision.is_none());
    }

    /// Every gap must name a concrete fix site; a report that says "not supported" and stops is
    /// the failure mode this whole path exists to replace.
    #[test]
    fn every_gap_names_a_fix_site() {
        let gaps = k3_gaps(&k3_cfg_from(&k3_json(&[])));
        assert!(
            gaps.len() >= 8,
            "expected the full ranked list, got {}",
            gaps.len()
        );
        for g in &gaps {
            assert!(
                g.fix.contains(".rs") || g.fix.contains(".h"),
                "gap {:?} names no file to change",
                g.what
            );
            assert!(!g.scope.is_empty() && !g.why.is_empty());
        }
    }

    /// A CLOSED capability must carry its evidence, and an OPEN one must not claim any.
    ///
    /// This is the assertion that keeps the report honest in both directions. Printing a landed
    /// capability as an unimplemented blocker sends the next agent to rebuild four opcodes that
    /// already dispatch (it did); marking one closed without the measured residual next to it
    /// makes the claim unfalsifiable.
    #[test]
    fn closed_gaps_carry_evidence_and_open_gaps_do_not_claim_any() {
        let gaps = k3_gaps(&k3_cfg_from(&k3_json(&[])));
        let closed: Vec<_> = gaps.iter().filter(|g| g.done.is_some()).collect();
        let open: Vec<_> = gaps.iter().filter(|g| g.done.is_none()).collect();
        assert!(
            closed.len() >= 5,
            "KDA, situ, AttnRes, LatentMoE, the MLA output gate and NoPE all landed with \
             real-weight gates; got {} closed",
            closed.len()
        );
        for g in &closed {
            // "validated" means a number, not an adjective.
            assert!(
                g.done.unwrap().contains("e-0") || g.done.unwrap().contains("e+0"),
                "closed gap {:?} cites no measured residual",
                g.what
            );
        }
        assert!(!open.is_empty(), "the full-model emit is still open");
        assert!(
            open[0].what.contains("full-model emit"),
            "the model-level assembly is THE remaining blocker and must rank first among the \
             open items; got {:?}",
            open[0].what
        );
    }

    /// The K3 config parse must NOT route through `require_mla_rope`.
    ///
    /// `require_mla_rope` lives on the `cfg_glm` path. A reader who sees it refuse NoPE naturally
    /// concludes it is what blocks the 93-layer K3 emit and "opens it for K3" — which changes
    /// nothing at all, because `k3_cfg_from` never calls it. Pin the fact so the next reader is
    /// not sent to the wrong file: K3's refusal comes from `kimi_k3_emit`, and the thing behind it
    /// is the absent model-level emitter.
    #[test]
    fn k3_is_refused_by_the_gap_report_not_by_require_mla_rope() {
        // A NoPE K3 config parses CLEANLY here. If `require_mla_rope` were on this path, this
        // call would panic instead of returning.
        let c = k3_cfg_from(&k3_json(&[]));
        assert!(c.mla_nope && c.rope_theta.is_none());
        // And the NoPE entry is CLOSED as a technique, not open as a blocker.
        let nope = k3_gaps(&c)
            .into_iter()
            .find(|g| g.what.contains("NO positional"))
            .expect("a NoPE model must still produce an explicit entry");
        assert!(
            nope.done.is_some(),
            "rung 3 proved the identity-table technique"
        );
        assert!(
            nope.fix.contains("require_mla_rope"),
            "the entry must say, in the fix text, that opening require_mla_rope is NOT the fix"
        );
    }

    /// The top-k gap is a real threshold against `PLOW_MOE_MAX_TOPK 8u`, not a blanket "kimi is
    /// unsupported": a top-8 config must NOT raise it. The clamp it guards
    /// (`runtime/amd/op_moe.h:135`) is silent, so this is the one gap whose absence would be
    /// indistinguishable from correctness at runtime.
    #[test]
    fn topk_gap_is_conditional_on_the_kernel_bound() {
        let has = |c: &K3Cfg| k3_gaps(c).iter().any(|g| g.what.contains("top-k beyond"));
        // Against the constant, never a literal — the bound moved 8 -> 16 for this very model,
        // and a hardcoded test would then have asserted the opposite of what it means.
        let over = (crate::MOE_MAX_TOPK + 1).to_string();
        let at = crate::MOE_MAX_TOPK.to_string();
        assert!(has(&k3_cfg_from(&k3_json(&[(
            "text_config/num_experts_per_token",
            &over
        )]))));
        assert!(!has(&k3_cfg_from(&k3_json(&[(
            "text_config/num_experts_per_token",
            &at
        )]))));
    }

    /// K3's real top-16 is now INSIDE the bound, so the gap must be gone from the shipped report.
    /// This is the assertion that the raise actually removed a blocker rather than renaming one.
    #[test]
    fn kimi_k3_real_topk_is_within_the_raised_bound() {
        assert!(16 <= crate::MOE_MAX_TOPK, "K3 routes top-16");
        let c = k3_cfg_from(&k3_json(&[("text_config/num_experts_per_token", "16")]));
        assert!(!k3_gaps(&c).iter().any(|g| g.what.contains("top-k beyond")));
    }

    /// The latent-MoE gap must fire only when the routed experts really do read a different
    /// width; a hidden-width MoE (DeepSeek/GLM shape) is already covered by the existing emit.
    #[test]
    fn latent_moe_gap_is_conditional_on_the_width() {
        let has = |c: &K3Cfg| k3_gaps(c).iter().any(|g| g.what.contains("LATENT MoE"));
        assert!(has(&k3_cfg_from(&k3_json(&[]))));
        assert!(!has(&k3_cfg_from(&k3_json(&[(
            "text_config/routed_expert_hidden_size",
            "256" // == hidden_size
        )]))));
    }
}
