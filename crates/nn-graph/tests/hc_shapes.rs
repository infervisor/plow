//! Shape inference for DeepSeek-V4.1's mHC ops.
//!
//! The mHC stream is `[B, S, hc_mult, H]` — one axis more than every other
//! residual in this IR. The rules below exist because a wrong wiring here does
//! not crash: the copy axis and the hidden axis are both just axes, and a
//! `pre_mix` that still carries hidden would broadcast into a plausible,
//! wrong model rather than fail.

use nn_graph::{infer_shapes, DType, Dim, Nn};

const H: i64 = 64;
const HC: u32 = 4;
/// `(2 + hc_mult) * hc_mult` — pre (4) + post (4) + the flattened 4x4 comb.
const MIX: i64 = 24;

/// Build `[B, S, hc, H]` as a graph input, plus the builder it came from.
fn stream() -> (Nn, nn_graph::TensorId, Dim, Dim) {
    let mut nn = Nn::new(DType::BF16, DType::BF16);
    let b = nn.sym("B");
    let s = nn.sym("S");
    let x = nn.input(
        "stream",
        nn.shape([b.clone(), s.clone(), Dim::stat(HC as i64), Dim::stat(H)]),
        DType::BF16,
    );
    (nn, x, b, s)
}

#[test]
fn hc_mixes_packs_pre_post_and_comb() {
    let (mut nn, x, _b, _s) = stream();
    let m = nn.hc_mixes("layers.0.hc_attn", x, H, HC, 20, 1e-6);
    nn.mark_output(m);
    let mut g = nn.finish();
    infer_shapes(&mut g).expect("shapes infer");

    let out = g.tensor(m).shape.clone().expect("shape");
    // The copy axis is consumed; hidden is replaced by the packed mixes.
    assert_eq!(out.rank(), 3, "{out}");
    assert_eq!(out.dim(2).as_static(), Some(MIX), "{out}");
}

/// `hc_fn` reads the stream FLATTENED over copies, so its input width is
/// `hc_mult * hidden`. Declaring it at `hidden` is the easy mistake and it is
/// rejected.
#[test]
fn hc_mixes_rejects_a_projection_sized_for_the_wrong_hidden() {
    let (mut nn, x, _b, _s) = stream();
    // The stream is [.., HC, 64]; declaring hidden as 32 sizes hc_fn at
    // HC * 32 instead of HC * 64. That is the same arithmetic error as
    // declaring it at `hidden` rather than `hc_mult * hidden`.
    let m = nn.hc_mixes("layers.0.hc_attn", x, H / 2, HC, 20, 1e-6);
    nn.mark_output(m);
    let mut g = nn.finish();
    let err = infer_shapes(&mut g).expect_err("an hc_fn sized for the wrong hidden is rejected");
    assert!(format!("{err}").contains("hc_fn"), "{err}");
}

#[test]
fn hc_pre_collapses_the_copy_axis() {
    let (mut nn, x, b, s) = stream();
    let pre = nn.input(
        "pre_mix",
        nn.shape([b, s, Dim::stat(HC as i64)]),
        DType::F32,
    );
    let y = nn.hc_pre(x, pre, HC);
    nn.mark_output(y);
    let mut g = nn.finish();
    infer_shapes(&mut g).expect("shapes infer");

    let out = g.tensor(y).shape.clone().expect("shape");
    assert_eq!(out.rank(), 3, "{out}");
    assert_eq!(out.dim(2).as_static(), Some(H), "{out}");
}

/// A `pre_mix` that still carries the hidden axis is the single-pass wiring
/// done wrong. It must fail rather than broadcast.
#[test]
fn hc_pre_rejects_a_pre_mix_with_hidden() {
    let (mut nn, x, b, s) = stream();
    let bad = nn.input(
        "pre_mix",
        nn.shape([b, s, Dim::stat(HC as i64), Dim::stat(H)]),
        DType::F32,
    );
    let y = nn.hc_pre(x, bad, HC);
    nn.mark_output(y);
    let mut g = nn.finish();
    let err = infer_shapes(&mut g).expect_err("pre_mix must not carry hidden");
    assert!(format!("{err}").contains("hc_pre"), "{err}");
}

#[test]
fn hc_post_restores_the_copy_axis() {
    let (mut nn, resid, b, s) = stream();
    let x = nn.input(
        "sublayer_out",
        nn.shape([b.clone(), s.clone(), Dim::stat(H)]),
        DType::BF16,
    );
    let post = nn.input(
        "post_mix",
        nn.shape([b.clone(), s.clone(), Dim::stat(HC as i64)]),
        DType::F32,
    );
    let comb = nn.input(
        "comb_mix",
        nn.shape([b, s, Dim::stat(HC as i64 * HC as i64)]),
        DType::F32,
    );
    let y = nn.hc_post(x, resid, post, comb, HC);
    nn.mark_output(y);
    let mut g = nn.finish();
    infer_shapes(&mut g).expect("shapes infer");

    let out = g.tensor(y).shape.clone().expect("shape");
    assert_eq!(out.rank(), 4, "{out}");
    assert_eq!(out.dim(2).as_static(), Some(HC as i64), "{out}");
    assert_eq!(out.dim(3).as_static(), Some(H), "{out}");
}

/// `comb` is the FLATTENED `[hc, hc]` matrix. Passing it at `hc` — the shape
/// `post_mix` has — is rejected, because the two are otherwise interchangeable
/// at the call site.
#[test]
fn hc_post_rejects_an_unflattened_comb() {
    let (mut nn, resid, b, s) = stream();
    let x = nn.input(
        "sublayer_out",
        nn.shape([b.clone(), s.clone(), Dim::stat(H)]),
        DType::BF16,
    );
    let post = nn.input(
        "post_mix",
        nn.shape([b.clone(), s.clone(), Dim::stat(HC as i64)]),
        DType::F32,
    );
    let bad = nn.input(
        "comb_mix",
        nn.shape([b, s, Dim::stat(HC as i64)]),
        DType::F32,
    );
    let y = nn.hc_post(x, resid, post, bad, HC);
    nn.mark_output(y);
    let mut g = nn.finish();
    let err = infer_shapes(&mut g).expect_err("comb must be the flattened hc x hc matrix");
    assert!(format!("{err}").contains("comb_mix"), "{err}");
}

/// A stream whose copy axis disagrees with `hc_mult` is rejected — otherwise a
/// 2-copy stream would silently run through a 4-copy mix.
#[test]
fn hc_ops_reject_a_mismatched_copy_count() {
    let mut nn = Nn::new(DType::BF16, DType::BF16);
    let b = nn.sym("B");
    let s = nn.sym("S");
    let x = nn.input(
        "stream",
        nn.shape([b, s, Dim::stat(2), Dim::stat(H)]),
        DType::BF16,
    );
    let m = nn.hc_mixes("layers.0.hc_attn", x, H, HC, 20, 1e-6);
    nn.mark_output(m);
    let mut g = nn.finish();
    let err = infer_shapes(&mut g).expect_err("2 copies against hc_mult=4 must be rejected");
    assert!(format!("{err}").contains("residual copies"), "{err}");
}
