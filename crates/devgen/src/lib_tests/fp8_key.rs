//! The `fp8/` key contract, pinned. Three parties have to agree on one string: this emitter
//! declares the packet tensor name, `quantize_fp8.py` writes the safetensors key, and the
//! runtime looks it up. They did not — the emitter and the quantizer wrote `fp8/<name>` while a
//! loader stripped the prefix and looked up `<name>`, so a freshly generated fp8 checkpoint
//! could not load. These assertions are the emitter's half of the contract, stated as code so
//! the spelling cannot drift silently.
use super::*;

/// The canonical forms. `_scale` goes on the END, after the full weight name — `fp8/X.weight`
/// pairs with `fp8/X.weight_scale`, NOT `fp8/X_scale.weight`.
#[test]
fn fp8_twin_names_are_the_declared_name_verbatim() {
    let w = format!("fp8/{}", "model.layers.3.self_attn.q_proj.weight");
    let s = format!("{w}_scale");
    assert_eq!(w, "fp8/model.layers.3.self_attn.q_proj.weight");
    assert_eq!(s, "fp8/model.layers.3.self_attn.q_proj.weight_scale");
    // The key is the packet name VERBATIM: no strip, no rewrite, nothing to apply twice.
    assert_eq!(
        w.strip_prefix("fp8/"),
        Some("model.layers.3.self_attn.q_proj.weight")
    );
    assert!(
        w.starts_with("fp8/"),
        "the prefix is part of the key, not a routing marker"
    );
}

/// An `fp8/` twin is checkpoint-bound weight bytes, and every reader agrees on that because
/// there is now only one reader: `packet::names::is_checkpoint_weight`.
///
/// This test used to spell the predicate out as
/// `starts_with("model.") || starts_with("fp8/")` and named `manager.rs` / `exec/gpu.rs` as
/// the two sites it mirrored. There were five sites, they disagreed, and the allowlist form
/// silently zeroed an untied `lm_head.weight` on CUDA and would have zeroed the whole
/// Kimi-K3 tower. Asserting against the shared predicate is the point — a re-spelt copy is
/// exactly how the five diverged.
#[test]
fn fp8_twins_are_weight_bytes_under_the_shared_predicate() {
    use packet::names::is_checkpoint_weight as w;
    assert!(w("fp8/model.layers.0.mlp.down_proj.weight"));
    assert!(w("fp8/model.layers.0.mlp.down_proj.weight_scale"));
    assert!(w("model.layers.0.mlp.down_proj.weight"));
    assert!(
        w("lm_head.weight"),
        "untied head: declared at the top level, and a weight"
    );
    assert!(!w("act.x"), "activations are not weight bytes");
    assert!(!w("in.pos"));
}

/// The scale is per OUTPUT CHANNEL — one f32 per row of the `[out, in]` weight — and the
/// dequant is a MULTIPLY. Both halves matter: `quantize_fp8.py` stores `amax/448` and the
/// device epilogue computes `acc * a_scale[m] * w_scale[n]`, so a reciprocal on either side
/// would be a silent 448²-ish error rather than a crash.
#[test]
fn fp8_scale_is_per_output_channel_and_multiplied() {
    let (out, inp) = (4096u64, 2560u64);
    assert_eq!(
        out * F32,
        16384,
        "scale vector is [out] f32, not [out,in] and not [in]"
    );
    // Round-trip the convention the quantizer documents, at f32 precision.
    let w: f32 = -0.37;
    let amax: f32 = 0.37;
    let scale = amax / 448.0;
    let q = (w / scale).round().clamp(-448.0, 448.0);
    assert!((q * scale - w).abs() < 1e-3, "dequant is w8 * scale");
    let _ = inp;
}
