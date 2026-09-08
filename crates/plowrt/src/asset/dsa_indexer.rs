//! §A DSA lightning-indexer fp8 -> bf16 upcast, at bind.
//!
//! A COMPATIBILITY SHIM FOR EXACTLY TWO NAMED PROJECTIONS. It is not a dtype-coercion
//! layer and must not grow into one: every other block-fp8 tensor in a GLM checkpoint
//! stays fp8 all the way to the kernel, which dequantises per 128-K block in the
//! accumulator (`gemv_blk_rows_r`, runtime/amd/op_gemm.h). Upcasting those would double
//! their footprint and throw away the whole point of a block-fp8 blob.
//!
//! # Why these two are different
//!
//! `devgen::mla` declares `self_attn.indexer.wq_b.weight` and `self_attn.indexer.wk.weight`
//! as BF16 unconditionally, and that is deliberate: plow's indexer ops read them as bf16,
//! and the reference computes both projections in bf16 regardless of the rest of the
//! model's quantization — vLLM builds the fused `wk_weights_proj` with `quant_config=None`
//! and dequantises a checkpoint's fp8 `wk` into it at load ("FP8 wk weights are upcasted
//! to BF16 during loading to maintain fusion", `deepseek_v2.py`).
//!
//! GLM-5.3-FP8 stores both as `F8_E4M3` with a `[128,128]` `weight_scale_inv` grid:
//!
//! | tensor | payload | scale grid |
//! |---|---|---|
//! | `indexer.wq_b.weight` | F8_E4M3 `[4096, 2048]` | F32 `[32, 16]` |
//! | `indexer.wk.weight`   | F8_E4M3 `[128, 6144]`  | F32 `[1, 48]`  |
//!
//! so the blob's bf16 byte count is twice what the checkpoint holds and the bind died in
//! [`crate::asset::shard::slice_for`] with `replicated but the checkpoint has 8388608 B
//! and the blob declares 16777216 B` — after the whole ~200 GiB model had been uploaded.
//! Every GLM-5.3 blob therefore shipped `PLOW_GLM_DSA=0`.
//!
//! # Why it is HERE and not in the emitter
//!
//! An emit-time refusal was tried and reverted. `devgen` does not read checkpoint headers,
//! and the MoE expert encoding is not a proxy for the indexer's dtype: a block-fp8 GLM with
//! a genuinely bf16 indexer is a legitimate emit, and keying the refusal on `MoeEnc::Fp8Blk`
//! broke seven GLM and Kimi emit tests that construct exactly that pair. The dtype is a fact
//! about the CHECKPOINT, so the resolution is the only place that can see it.
//!
//! # The arithmetic
//!
//! `out[n][k] = bf16( f32(fp8[n][k]) * scale[n/128][k/128] )`, the block-index gather every
//! other consumer of this grid in the tree uses (`dev_isa.h`'s `S[(n>>7)*ceil(K/128) + (k>>7)]`,
//! `scripts/glm52_prep.py`'s `dequant_blockfp8`, vLLM's `scaled_dequantize` with
//! `GroupShape(128, 128)`).
//!
//! The e4m3 decode is EXACT — three mantissa bits fit in bf16's seven — so the only rounding
//! is the f32 multiply by an arbitrary-f32 block scale and the round-to-nearest-even
//! narrowing back to bf16, which is exactly what the reference does. This is a lossier
//! representation than the kernel's per-block dequant-in-the-accumulator, and that is
//! inherent to fusing the projection in bf16; it is what the reference ships.

use safetensors::Dtype;

use crate::asset::checkpoint::Checkpoint;
use crate::{Result, RuntimeError};

/// The two projections this shim covers. Exhaustive, and matched by SUFFIX so the
/// `model.layers.{l}.` prefix (or a wrapper's) does not have to be spelled here.
///
/// `.wk.weight` and not `wk`: the reference's fused parameter is spelled
/// `wk_weights_proj.weight`, and a bare `wk` test would catch it.
const PROJECTIONS: [&str; 2] = [
    "self_attn.indexer.wq_b.weight",
    "self_attn.indexer.wk.weight",
];

/// DeepSeek/GLM block-fp8 quantises on a `[128,128]` grid. Not a parameter anywhere in this
/// tree — `devgen::mla::mla_ckpt_enc` asserts `weight_block_size == [128,128]` at emit and
/// every scale-grid size is written as `div_ceil(128)` — so a checkpoint quantised at any
/// other block size is refused rather than silently mis-indexed.
const BLK: usize = 128;

/// Is `name` one of the two indexer projections?
pub fn is_indexer_projection(name: &str) -> bool {
    PROJECTIONS.iter().any(|s| name.ends_with(s))
}

/// What the loader must do with `name`, decided from the checkpoint alone.
///
/// Resolved once at preflight (before a byte is uploaded) and again at bind, from the same
/// function, so the two cannot come apart.
pub enum Plan<'a> {
    /// The checkpoint really is bf16. Bind its bytes unchanged — the ordinary path.
    AsIs,
    /// Block-fp8 on disk. Dequantise into bf16 with [`Plan::dequantise`].
    Upcast {
        w: &'a [u8],
        scale: &'a [u8],
        n: usize,
        k: usize,
        sk: usize,
    },
}

/// Decide what to do with `name`, or `None` when it is not one of the two projections.
///
/// `want` is the blob's declared byte count for this rank. Both projections are REPLICATED
/// (`shard::shard_of` classifies them so — the indexer is tiny and its index is head-shared),
/// so `want` is the whole bf16 tensor at every tp and the equality below is the real check.
///
/// Every refusal names the tensor, both dtypes and the remedy, because the alternative that
/// shipped was a `slice_for` byte-count error 200 GiB into a load.
pub fn plan<'a>(ckpt: &'a Checkpoint, name: &str, want: u64) -> Option<Result<Plan<'a>>> {
    if !is_indexer_projection(name) {
        return None;
    }
    Some(plan_inner(ckpt, name, want))
}

fn plan_inner<'a>(ckpt: &'a Checkpoint, name: &str, want: u64) -> Result<Plan<'a>> {
    let bad = |m: String| RuntimeError::Device(format!("DSA indexer {name}: {m}"));
    let (w, shape) = ckpt
        .tensor_ex(name)
        .ok_or_else(|| bad("not in the checkpoint".into()))?;

    // Not fp8 => the emitter's bf16 claim is true and there is nothing to do. Any size
    // disagreement in that case is an ordinary one and belongs to `slice_for`, which says
    // it better than this function could.
    if !ckpt.is_fp8_e4m3(name) {
        return Ok(Plan::AsIs);
    }

    // The whole reason this file exists: the blob says bf16, the checkpoint says fp8.
    // Refuse anything the shim cannot turn into exactly `want` bytes of bf16.
    let dt = ckpt.dtype(name).unwrap_or(Dtype::F8_E4M3);
    let remedy = "emit with PLOW_GLM_DSA=0, or supply a checkpoint whose indexer is bf16";
    if shape.len() != 2 {
        return Err(bad(format!(
            "checkpoint dtype {dt:?}, blob declares BF16, but the checkpoint shape is \
             {shape:?} — the block-fp8 upcast needs a 2-D [out, in] weight. {remedy}"
        )));
    }
    let (n, k) = (shape[0], shape[1]);
    if w.len() != n * k {
        return Err(bad(format!(
            "checkpoint dtype {dt:?} is {} B for shape {shape:?} — expected one byte per \
             element. {remedy}",
            w.len()
        )));
    }
    if want != (n * k * 2) as u64 {
        return Err(bad(format!(
            "checkpoint dtype {dt:?} {shape:?} upcasts to {} B of BF16 but the blob declares \
             {want} B. {remedy}",
            n * k * 2
        )));
    }

    let sname = format!("{name}_scale_inv");
    let (sbytes, sshape) = ckpt.tensor_ex(&sname).ok_or_else(|| {
        bad(format!(
            "checkpoint dtype {dt:?}, blob declares BF16, and the block-fp8 upcast needs \
             `{sname}` — which is not in the checkpoint. {remedy}"
        ))
    })?;
    if ckpt.dtype(&sname) != Some(Dtype::F32) {
        return Err(bad(format!(
            "`{sname}` is {:?}, and a DeepSeek/GLM block scale is F32 (ARBITRARY, not a \
             power of two — it cannot be an E8M0 microscaling row). {remedy}",
            ckpt.dtype(&sname)
        )));
    }
    // The grid must be EXACTLY the [ceil(N/128), ceil(K/128)] one the whole tree writes and
    // reads. Deriving the block size from the grid instead would accept a [64,64]-quantised
    // checkpoint and index it as if it were [128,128]: same byte count, plausible output,
    // and an indexer that selects the wrong KV rows — the failure no serving smoke test
    // catches, which is why this is checked rather than inferred.
    let (want_sn, want_sk) = (n.div_ceil(BLK), k.div_ceil(BLK));
    if sshape != [want_sn, want_sk] {
        return Err(bad(format!(
            "`{sname}` is {sshape:?} but a [{BLK},{BLK}] block-fp8 grid over {shape:?} is \
             [{want_sn}, {want_sk}]. Missing capability: `fp8_block_size_{sshape:?}`. {remedy}"
        )));
    }
    if sbytes.len() != want_sn * want_sk * 4 {
        return Err(bad(format!(
            "`{sname}` is {} B for shape {sshape:?} — expected {} B of F32. {remedy}",
            sbytes.len(),
            want_sn * want_sk * 4
        )));
    }
    Ok(Plan::Upcast {
        w,
        scale: sbytes,
        n,
        k,
        sk: want_sk,
    })
}

impl Plan<'_> {
    /// The bf16 bytes to bind, or `None` for [`Plan::AsIs`] (nothing to build).
    ///
    /// Only reached at bind, never at preflight: [`plan`] validates without touching a page
    /// of the payload, so the preflight costs a hash lookup per tensor and no faults.
    pub fn dequantise(&self) -> Option<Vec<u8>> {
        let &Plan::Upcast { w, scale, n, k, sk } = self else {
            return None;
        };
        // 256-entry decode table. e4m3 -> bf16 is EXACT (3 mantissa bits into 7), so this
        // holds every value the format has and the f32 widening below is lossless; the only
        // rounding in the whole function is the RNE narrowing after the scale multiply.
        let lut = e4m3_f32_table();
        let mut out = vec![0u8; n * k * 2];
        for row in 0..n {
            let srow = (row / BLK) * sk;
            let (wr, or) = (row * k, row * k * 2);
            for kb in 0..sk {
                let s = f32::from_le_bytes(
                    scale[(srow + kb) * 4..(srow + kb) * 4 + 4]
                        .try_into()
                        .expect("4 B of f32"),
                );
                let (lo, hi) = (kb * BLK, ((kb + 1) * BLK).min(k));
                for c in lo..hi {
                    let v = f32_to_bf16_bits(lut[w[wr + c] as usize] * s);
                    out[or + c * 2..or + c * 2 + 2].copy_from_slice(&v.to_le_bytes());
                }
            }
        }
        Some(out)
    }
}

/// OCP e4m3 (`F8_E4M3`, a.k.a. e4m3fn) byte -> f32, all 256 encodings.
///
/// The software statement of the format, transliterated from `plow_fp8_ocp_to_bf16`
/// (`runtime/amd/amd_arch.h`) — NOT from `plow_fp8x4_ocp_to_bf16`, whose `* 2.0f` is a
/// gfx942 artifact (its hardware cvt reads e4m3FNUZ, and `OCP(b) == 2 * FNUZ(b)`). A pure
/// software decode has no such factor.
///
/// `0x80` is OCP `-0.0` and stays `-0.0`: the `0x80` scrub the upload ring applies exists
/// only because that byte is FNUZ NaN to the hardware cvt, and nothing downstream of this
/// function sees an fp8 byte. Its bf16 is `0x8000`, which is `-0.0` and behaves as zero in
/// every GEMM that reads it.
fn e4m3_f32_table() -> [f32; 256] {
    let mut t = [0.0f32; 256];
    for (b, slot) in t.iter_mut().enumerate() {
        let sign = if b & 0x80 != 0 { -1.0f32 } else { 1.0 };
        let (e, m) = ((b >> 3) & 0x0F, b & 0x07);
        *slot = if e == 0 {
            // Subnormal: m * 2^-9.
            sign * (m as f32) * (1.0 / 512.0)
        } else if e == 15 && m == 7 {
            // The one OCP NaN encoding (e4m3fn has no infinities).
            f32::NAN
        } else {
            // exponent bias 7; the implicit leading 1 is present.
            sign * (1.0 + m as f32 / 8.0) * exp2i(e as i32 - 7)
        };
    }
    t
}

/// `2^e` for the exponents e4m3 can hold (-6..=8), by f32 bit construction — no libm on a
/// table this small, and exact by definition.
fn exp2i(e: i32) -> f32 {
    f32::from_bits((((e + 127) as u32) & 0xFF) << 23)
}

/// f32 -> bf16 bits, round-to-nearest-even — the same rounding `torch.Tensor.bfloat16()`
/// applies, which is what makes the host verification in `scripts/glm53_dsa_verify.py`
/// a bit-for-bit comparison rather than a tolerance one.
fn f32_to_bf16_bits(x: f32) -> u16 {
    let u = x.to_bits();
    if x.is_nan() {
        // Quiet the NaN rather than let the round turn it into an infinity.
        return ((u >> 16) as u16) | 0x0040;
    }
    ((u.wrapping_add(0x7FFF + ((u >> 16) & 1))) >> 16) as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Only the two projections, and not the reference's fused spelling — which is a
    /// DIFFERENT tensor (`[160, 6144]` = wk stacked on weights_proj) that plow never binds.
    #[test]
    fn only_the_two_indexer_projections_match() {
        assert!(is_indexer_projection(
            "model.layers.0.self_attn.indexer.wq_b.weight"
        ));
        assert!(is_indexer_projection(
            "model.layers.77.self_attn.indexer.wk.weight"
        ));
        for n in [
            "model.layers.0.self_attn.indexer.wk_weights_proj.weight",
            "model.layers.0.self_attn.indexer.weights_proj.weight",
            "model.layers.0.self_attn.indexer.k_norm.weight",
            "model.layers.0.self_attn.indexer.wq_b.weight_scale_inv",
            "model.layers.0.self_attn.q_b_proj.weight",
            "model.layers.0.mlp.down_proj.weight",
        ] {
            assert!(!is_indexer_projection(n), "{n} must not be upcast");
        }
    }

    /// Exhaustive against the format: every finite e4m3 encoding round-trips through bf16
    /// unchanged, which is the property that makes the scale multiply the only lossy step.
    #[test]
    fn every_e4m3_encoding_is_exact_in_bf16() {
        let t = e4m3_f32_table();
        for b in 0..256usize {
            let v = t[b];
            if v.is_nan() {
                assert_eq!(b & 0x7F, 0x7F, "only 0x7f/0xff are NaN, not {b:#04x}");
                continue;
            }
            let back = f32::from_bits((f32_to_bf16_bits(v) as u32) << 16);
            assert_eq!(v.to_bits(), back.to_bits(), "{b:#04x} = {v} lost bits in bf16");
        }
        // Anchors, read off the format rather than off this implementation.
        assert_eq!(t[0x00], 0.0);
        assert_eq!(t[0x08], 2f32.powi(-6)); // smallest normal
        assert_eq!(t[0x01], 2f32.powi(-9)); // smallest subnormal
        assert_eq!(t[0x7E], 448.0); // largest finite
        assert_eq!(t[0x38], 1.0);
        assert_eq!(t[0xB8], -1.0);
        assert!(t[0x80].is_sign_negative() && t[0x80] == 0.0); // OCP -0.0, not NaN
    }

    /// Dequantise the real checkpoint's indexer projections and write them out for
    /// `scripts/glm53_dsa_verify.py` to compare against torch.
    ///
    /// Opt-in, because it needs a ~200 GiB checkpoint:
    ///
    /// ```text
    /// PLOW_DSA_VERIFY_CKPT=/workspace/models/GLM-5.3-plow-lite \
    /// PLOW_DSA_VERIFY_OUT=/tmp/dsa \
    ///   cargo test --release -p plowrt --features hsa --lib -- --ignored dsa_indexer
    /// ```
    ///
    /// A silent scale-grid indexing error here is the failure mode worth this much
    /// machinery: it produces a plausible-looking indexer of exactly the right size that
    /// selects the WRONG KV rows, and no serving smoke test would catch it.
    #[test]
    #[ignore = "needs a real block-fp8 GLM checkpoint (PLOW_DSA_VERIFY_CKPT)"]
    fn upcast_the_real_checkpoint_for_the_host_comparison() {
        let Ok(dir) = std::env::var("PLOW_DSA_VERIFY_CKPT") else {
            panic!("set PLOW_DSA_VERIFY_CKPT to a checkpoint directory");
        };
        let out = std::path::PathBuf::from(
            std::env::var("PLOW_DSA_VERIFY_OUT").unwrap_or_else(|_| "/tmp/dsa".into()),
        );
        std::fs::create_dir_all(&out).expect("create the output directory");
        let ckpt = Checkpoint::open(std::path::Path::new(&dir)).expect("open the checkpoint");

        let mut n_done = 0usize;
        for l in [0usize, 1, 7, 40, 77] {
            for leaf in PROJECTIONS {
                let name = format!("model.layers.{l}.{leaf}");
                let Some((_, shape)) = ckpt.tensor_ex(&name) else {
                    continue;
                };
                // `want` is what devgen declares: the bf16 byte count.
                let want = (shape.iter().product::<usize>() * 2) as u64;
                let p = plan(&ckpt, &name, want)
                    .unwrap_or_else(|| panic!("{name} is not classified as an indexer projection"))
                    .unwrap_or_else(|e| panic!("{name}: {e}"));
                let Some(bytes) = p.dequantise() else {
                    // bf16 on disk already — nothing for the host to check.
                    continue;
                };
                assert_eq!(bytes.len() as u64, want, "{name}");
                std::fs::write(out.join(format!("{name}.bf16")), &bytes).expect("write");
                n_done += 1;
            }
        }
        assert!(
            n_done > 0,
            "no block-fp8 indexer projection in {dir} — nothing was verified"
        );
        eprintln!("wrote {n_done} upcast tensors to {}", out.display());
    }

    /// RNE, not truncation: the halfway case must go to the even bf16.
    #[test]
    fn bf16_narrowing_rounds_to_nearest_even() {
        // 1.0 + 2^-8 is exactly halfway between bf16 1.0 (mantissa 0x00, even) and the next
        // bf16 up (0x01), so it must round DOWN to 1.0.
        assert_eq!(f32_to_bf16_bits(f32::from_bits(0x3F80_8000)), 0x3F80);
        // 1.0 + 3*2^-8 is halfway between 0x3F81 (odd) and 0x3F82 (even) -> up.
        assert_eq!(f32_to_bf16_bits(f32::from_bits(0x3F81_8000)), 0x3F82);
        // Truncation would give 0x3F80 for both.
        assert_eq!(f32_to_bf16_bits(1.0), 0x3F80);
        assert_eq!(f32_to_bf16_bits(-1.0), 0xBF80);
    }
}
