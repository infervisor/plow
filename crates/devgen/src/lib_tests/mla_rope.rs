use serde_json::{json, Value};

/// `cfg_glm`'s theta lookup and its refusal, extracted so both can be exercised without a
/// checkpoint on disk. MUST stay identical to the three lines in `mla::cfg_glm`.
fn resolve(v: &Value) -> Option<f64> {
    let rp = &v["rope_parameters"];
    let theta = v["rope_theta"]
        .as_f64()
        .or_else(|| rp["rope_theta"].as_f64());
    super::require_mla_rope(
        theta,
        v["mla_use_nope"].as_bool().unwrap_or(false),
        rp["rope_type"].as_str(),
        v["rope_scaling"].as_object().is_some(),
        v["model_type"].as_str().unwrap_or("<test>"),
    );
    theta
}

/// The SHIPPING model's spelling. GLM-5.2's `config.json` has NO top-level `rope_theta`; it
/// carries `rope_parameters: {rope_theta, rope_type}` (transformers 5.x moved the key). The
/// old `.unwrap_or(8_000_000.0)` therefore never read GLM's theta at all — it matched only
/// because the literal in `mla.rs` happened to equal it.
///
/// Asserted against the value IN the fixture, so a fixture edit cannot leave this passing
/// while the parse reads nothing.
#[test]
fn the_theta_comes_from_rope_parameters_not_from_a_default() {
    let v = json!({
        "model_type": "glm_moe_dsa",
        "rope_parameters": { "rope_theta": 8_000_000.0, "rope_type": "default" },
    });
    assert_eq!(resolve(&v), v["rope_parameters"]["rope_theta"].as_f64());
    // A different theta under the same spelling must produce that theta, not GLM's. This is
    // the property the default destroyed: every model read as 8e6, and all of them looked
    // right as long as they were GLM.
    let other = json!({
        "model_type": "some_other_mla",
        "rope_parameters": { "rope_theta": 123_457.0, "rope_type": "default" },
    });
    assert_eq!(
        resolve(&other),
        other["rope_parameters"]["rope_theta"].as_f64()
    );
    assert_ne!(
        resolve(&other),
        resolve(&v),
        "two configs must not resolve to one theta"
    );
}

/// The flat spelling still works and takes precedence.
#[test]
fn the_top_level_spelling_is_still_read() {
    let v = json!({ "model_type": "deepseek_v3", "rope_theta": 10_000.0 });
    assert_eq!(resolve(&v), v["rope_theta"].as_f64());
}

/// Kimi-K3: `mla_use_nope: true`, no theta anywhere. VERIFIED against the checkpoint —
/// `config.json`'s only `rope`-ish key is `text_config.qk_rope_head_dim`, and
/// `modeling_kimi_linear.py` has `self.rotary_emb = None` / `assert self.use_nope`.
#[test]
#[should_panic(expected = "mla_use_nope")]
fn a_nope_model_is_refused_not_given_glms_theta() {
    resolve(&json!({ "model_type": "kimi_k3", "mla_use_nope": true }));
}

/// The refusal names the consequence, not just the flag — a NoPE emit is not "delete the two
/// HeadNormRope ops", because the k-side one is the only writer of the krot cache row.
#[test]
fn the_nope_refusal_names_the_krot_cache() {
    let msg = std::panic::catch_unwind(|| {
        resolve(&json!({ "model_type": "kimi_k3", "mla_use_nope": true }))
    })
    .unwrap_err();
    let msg = msg
        .downcast_ref::<String>()
        .cloned()
        .unwrap_or_else(|| msg.downcast_ref::<&str>().map(|s| s.to_string()).unwrap());
    assert!(
        msg.contains("krot"),
        "refusal must name the dangling cache write; got: {msg}"
    );
}

/// Contradiction: a theta AND `mla_use_nope`. One of the two is wrong and the compiler
/// cannot tell which, so it refuses instead of picking.
#[test]
#[should_panic(expected = "contradict")]
fn a_theta_alongside_use_nope_is_a_contradiction() {
    resolve(&json!({
        "model_type": "confused", "mla_use_nope": true, "rope_theta": 8_000_000.0,
    }));
}

/// No theta, no NoPE flag: the compiler does not know the model's positional encoding.
/// This is the case the default silently answered with GLM's number.
#[test]
#[should_panic(expected = "no RoPE theta")]
fn an_absent_theta_is_a_refusal_not_eight_million() {
    resolve(&json!({ "model_type": "mystery_mla" }));
}

/// `declare_glm` builds its tables with `RopeScale::None`, so a scaled scheme would be
/// emitted as an UNSCALED RoPE at the base theta — right-looking tables, wrong long context.
#[test]
#[should_panic(expected = "rope_type")]
fn a_scaled_rope_scheme_is_refused_rather_than_silently_unscaled() {
    resolve(&json!({
        "model_type": "yarned",
        "rope_parameters": { "rope_theta": 500_000.0, "rope_type": "yarn" },
    }));
}

/// The legacy `rope_scaling` object, same reason.
#[test]
#[should_panic(expected = "rope_scaling")]
fn a_legacy_rope_scaling_object_is_refused() {
    resolve(&json!({
        "model_type": "scaled",
        "rope_theta": 500_000.0,
        "rope_scaling": { "type": "linear", "factor": 4.0 },
    }));
}

/// `require_mla_geometry` refuses every MLA shape no kernel in this tree is instantiated for.
///
/// The failure it exists to prevent is the worst class in the emitter, because it is not a
/// missing arm — which on AMD merely writes nothing — but a PRESENT arm reading a shape it was
/// not built for: every MLA dispatch site hardcodes `<512, ...>`, so a `kv_lora_rank` of 384
/// would emit correctly-shaped tensors and have the kernel read 512 elements out of each
/// 384-wide latent row, past the end of the cache.
#[test]
fn an_unsupported_mla_geometry_is_refused_at_the_config_parse() {
    let bad = |dk: u32, dr: u32, vd: u32| -> String {
        let e = std::panic::catch_unwind(move || super::require_mla_geometry(dk, dr, vd, "probe"))
            .expect_err("must refuse");
        e.downcast_ref::<String>()
            .cloned()
            .or_else(|| e.downcast_ref::<&str>().map(|s| (*s).to_string()))
            .unwrap_or_default()
    };

    // The two geometries that ARE instantiated, both of which must pass.
    super::require_mla_geometry(512, 64, 128, "deepseek/glm/k2.7");
    super::require_mla_geometry(512, 0, 256, "nope");

    // A latent width no kernel is built for. The message must name the value, the constant, and
    // WHY it is not a config knob — an operator who reads only "unsupported" will try to make it
    // supported by editing the config.
    let m = bad(384, 64, 128);
    assert!(m.contains("384") && m.contains("512"), "names both widths: {m}");
    assert!(m.contains("TEMPLATE ARGUMENT"), "says why it is fixed: {m}");
    assert!(m.contains("op_attention.h"), "names the fix site: {m}");

    // An uninstantiated rope width. Distinguished from the DK case on purpose: DR is general in
    // the body (the zero-rope arm made it so), so this refusal says "not qualified", not
    // "impossible", and points at the coverage that would qualify it.
    let m = bad(512, 96, 128);
    assert!(m.contains("96"), "names the value: {m}");
    assert!(m.contains("mla_gfx950_test.c"), "names the coverage to add: {m}");

    // A v_head_dim that makes the fold's fast map unreachable in every arm. Correct output, 7.7x
    // slower, and completely silent about it — which is why it is refused rather than tolerated.
    for vd in [0u32, 2, 130] {
        let m = bad(512, 64, vd);
        assert!(
            m.contains("PLOW_MLA_FOLD_VEC"),
            "v_head_dim={vd} must name the constant it violates: {m}"
        );
    }
    // ...and the multiples of 4 that surround them are accepted.
    for vd in [4u32, 128, 132, 256] {
        super::require_mla_geometry(512, 64, vd, "probe");
    }
}
