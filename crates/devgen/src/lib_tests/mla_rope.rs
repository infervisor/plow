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
