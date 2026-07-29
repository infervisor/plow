//! Megatron weight sharding — which slice of a checkpoint tensor rank `r` owns.
//!
//! `plans/tp-design.md` §3a pairs a **column-parallel** producer with a
//! **row-parallel** consumer so each sublayer needs exactly one all-reduce and
//! no all-gather. That makes the bind a two-case problem:
//!
//! | shard | matrices | slice of `[out, in]` | cost |
//! |---|---|---|---|
//! | [`Shard::Column`] | q, k, v, gate, up | a contiguous ROW range | a byte offset |
//! | [`Shard::Row`] | o, down | a strided COLUMN range | a gather |
//! | [`Shard::Replicated`] | norms, embed, **lm_head** | all of it | nothing |
//!
//! # lm_head is REPLICATED unless the PACKET says otherwise
//!
//! §3a's table says lm_head is vocab-column-parallel with an `XArgmaxFin` to
//! fold the per-rank maxima. That op used to be a **stub**, so `crates/devgen`
//! kept lm_head full-vocab under TP and every rank argmaxed the whole vocabulary
//! independently; sharding it here would have quartered the logits and folded
//! nothing, and the ranks would have disagreed on the first token.
//!
//! `d_xargmax_fin_mega` (`runtime/amd/op_collective.h`) now exists and devgen's
//! `GLM_SHARD_HEAD=1` emits the column-parallel arm. So lm_head is the ONE
//! tensor whose shard is a property of the packet rather than a fixed rule, and
//! [`slice_for`] decides it the only way that cannot drift from the emitter:
//! from the DECLARED size. `full == want` means the packet asked for the whole
//! table (replicated); `full == want * tp` means it asked for a 1/tp slice. A
//! packet built without the knob is byte-for-byte unaffected.
//!
//! # Why the column shard is not simply `rank`
//!
//! Under GQA a model can have fewer KV heads than ranks — Gemma-4's full layers
//! have 4, so at TP=8 `k_proj`/`v_proj` cannot split eight ways. `devgen` then
//! declares a shard that divides the matrix into FEWER than `tp` pieces and has
//! `tp/n_shards` ranks **share** each piece (§3a's kv-head replication). So the
//! piece index is `rank / (tp / n_shards)`, and `n_shards` is recovered from the
//! declaration itself: `full_bytes / shard_bytes`. For q/gate/up `n_shards ==
//! tp` and the fold collapses back to `rank`.

use std::borrow::Cow;

use crate::{Result, RuntimeError};

/// Which axis of a weight TP splits, if any.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shard {
    /// Every rank binds the whole tensor.
    Replicated,
    /// Output-dim split: a contiguous row range of `[out, in]`.
    Column,
    /// Input-dim split: `in/N` contiguous elements out of every row.
    Row,
}

/// Classify a checkpoint tensor by name.
///
/// Substring tests on the HF name, matching `runtime/tests/tp_decode.c`. Keying
/// on the name is what lets the loader stage a sub-range of the ORIGINAL
/// checkpoint rather than requiring a re-sharded one on disk (§3b(b)).
pub fn shard_of(name: &str) -> Shard {
    // The last three are GLM-5.2 / MLA (`GlmMoeDsa`), whose attention has no q/k/v_proj at all:
    // the prep (`scripts/glm52_prep.py`) writes ABSORBED weights instead. They are column-parallel
    // for the same reason q_proj is -- the split axis is the HEAD count:
    //     derived.q_absorb [NH*DK, QL]   derived.q_rope [NH*DR, QL]   derived.v_absorb [NH*DK, VD]
    // `derived.kv_a_latent` [DK,H] and `derived.k_rope` [DR,H] are deliberately ABSENT: the latent
    // KV path is shared by every head, so there is no head axis to cut and every rank binds it
    // whole. This list is exactly `glm_col()` in `runtime/tests/glm52_decode.c`, which is the
    // arrangement the TP4 decode was proven against.
    //
    // GLM's remaining projections need no new entries and must not get any: the block-fp8 SCALE
    // twins ride along on the substring test, because "down_proj.weight_scale_inv" contains
    // "down_proj.weight". That is correct -- a [N/128, K/128] scale grid shards on the same axis
    // as the weight it scales, and the `shape.len() != 2` guard below leaves 1-D scales alone.
    const COL: [&str; 8] = [
        "q_proj.weight",
        "k_proj.weight",
        "v_proj.weight",
        "gate_proj.weight",
        "up_proj.weight",
        "derived.q_absorb",
        "derived.q_rope",
        "derived.v_absorb",
    ];
    const ROW: [&str; 2] = ["o_proj.weight", "down_proj.weight"];
    if COL.iter().any(|s| name.contains(s)) {
        Shard::Column
    } else if ROW.iter().any(|s| name.contains(s)) {
        Shard::Row
    } else {
        // Norms, embed_tokens, and lm_head — see the module note on lm_head.
        Shard::Replicated
    }
}

/// This rank's slice of `src`, given what the blob declared it should be.
///
/// `want` is the blob's declared byte size — the shard size, which is what makes
/// `n_shards` recoverable. `shape` is the checkpoint's, needed only by
/// [`Shard::Row`].
///
/// Borrows for column and replicated (a sub-range of the mmap); allocates only
/// for row, which is a genuine gather.
pub fn slice_for<'a>(
    name: &str,
    src: &'a [u8],
    shape: &[usize],
    want: u64,
    rank: u32,
    tp: u32,
) -> Result<Cow<'a, [u8]>> {
    let bad = |m: String| RuntimeError::Device(format!("shard {name}: {m}"));
    let full = src.len() as u64;

    // tp==1 binds everything whole, which is byte-identical to the pre-TP path.
    let shard = if tp > 1 { shard_of(name) } else { Shard::Replicated };

    // A row-parallel SCALE is replicated. An fp8 per-output-channel scale is
    // `[out]` — 1-D — and row-parallel splits the INPUT axis, which such a
    // tensor does not have. Its twin weight is sharded; the scale is not
    // (`tp_decode.c` §fp8: "its [out] scale is NOT sharded"). Keying on the
    // dimensionality rather than on the name says why: there is no axis to cut.
    let shard = match shard {
        Shard::Row if shape.len() != 2 => Shard::Replicated,
        s => s,
    };

    // lm_head: vocab-column-parallel iff the packet declared a 1/tp slice (devgen
    // `GLM_SHARD_HEAD`, folded across ranks by `XARGMAX_FIN`). Decided from the
    // declared size, not from a name list or a host flag, so this table cannot
    // disagree with the emitter — the failure mode the module note describes is a
    // silently wrong first token, and there is no host-side signal that would
    // catch it. `full == want` (the default) keeps the replicated path exactly.
    let shard = match shard {
        Shard::Replicated
            if name.ends_with("lm_head.weight") && want != 0 && full == want * tp as u64 =>
        {
            Shard::Column
        }
        s => s,
    };

    match shard {
        Shard::Replicated => {
            if full != want {
                return Err(bad(format!(
                    "replicated but the checkpoint has {full} B and the blob declares {want} B"
                )));
            }
            Ok(Cow::Borrowed(src))
        }

        Shard::Column => {
            // Contiguous row range. `n_shards` may be < tp when KV heads are
            // replicated across ranks — see the module note.
            if want == 0 || full % want != 0 {
                return Err(bad(format!(
                    "column-parallel, but the full {full} B is not a multiple of the \
                     declared shard {want} B"
                )));
            }
            let n_shards = full / want;
            if n_shards == 0 || tp as u64 % n_shards != 0 {
                return Err(bad(format!(
                    "column-parallel into {n_shards} shards, which does not divide tp={tp} \
                     (kv-head replication needs tp % n_shards == 0)"
                )));
            }
            let idx = rank as u64 / (tp as u64 / n_shards);
            let off = (idx * want) as usize;
            Ok(Cow::Borrowed(&src[off..off + want as usize]))
        }

        Shard::Row => {
            let (out, in_full) = (shape[0] as u64, shape[1] as u64);
            if out == 0 || in_full == 0 {
                return Err(bad(format!("row-parallel with a degenerate shape {shape:?}")));
            }
            if in_full % tp as u64 != 0 {
                return Err(bad(format!(
                    "row-parallel but the input dim {in_full} is not divisible by tp={tp}"
                )));
            }
            // Element width comes from the bytes, not from a precision flag: the
            // same gather serves bf16 (2 B) and fp8 e4m3 (1 B), and a flag is
            // one more thing that can disagree with the data.
            let elems = out * in_full;
            if elems == 0 || full % elems != 0 {
                return Err(bad(format!(
                    "row-parallel: {full} B does not divide into {out}x{in_full} elements"
                )));
            }
            let esz = full / elems;
            let in_sh = in_full / tp as u64;
            if out * in_sh * esz != want {
                return Err(bad(format!(
                    "row-parallel slice is {out}x{in_sh}x{esz} = {} B but the blob \
                     declares {want} B",
                    out * in_sh * esz
                )));
            }
            let row_dst = (in_sh * esz) as usize;
            let row_src = (in_full * esz) as usize;
            let skip = (rank as u64 * in_sh * esz) as usize;
            let mut buf = vec![0u8; want as usize];
            for r in 0..out as usize {
                let s = r * row_src + skip;
                buf[r * row_dst..(r + 1) * row_dst].copy_from_slice(&src[s..s + row_dst]);
            }
            Ok(Cow::Owned(buf))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// lm_head is the one tensor whose shard is a PACKET property. `shard_of` —
    /// the NAME rule — must keep saying Replicated (that is the default emit and
    /// the only answer available from a name); `slice_for` promotes it to Column
    /// when the declared size says the packet asked for a 1/tp slice. Both
    /// directions are pinned, because getting either wrong is a silently wrong
    /// first token with no host-side signal.
    #[test]
    fn lm_head_shard_follows_the_declared_size() {
        const V: usize = 64;
        const H: usize = 8;
        // distinct per 64-byte block, so a wrong OFFSET fails and not just a wrong length
        let full: Vec<u8> = (0..V * H * 2).map(|i| (i / 64) as u8).collect();
        // replicated: the packet declares the whole table
        let want = (V * H * 2) as u64;
        let got = slice_for("lm_head.weight", &full, &[V, H], want, 2, 4).unwrap();
        assert_eq!(got.len(), want as usize);
        assert_eq!(shard_of("lm_head.weight"), Shard::Replicated);
        // sharded: the packet declares vocab/tp rows, so rank 2 gets the 3rd quarter
        let want = want / 4;
        let got = slice_for("lm_head.weight", &full, &[V, H], want, 2, 4).unwrap();
        assert_eq!(got.len(), want as usize);
        assert_eq!(&got[..], &full[2 * want as usize..3 * want as usize]);
        // tp==1 never shards anything
        let got = slice_for("lm_head.weight", &full, &[V, H], full.len() as u64, 0, 1).unwrap();
        assert_eq!(got.len(), full.len());
    }

    /// The classification is the whole contract: the NAME rule keeps lm_head and
    /// embed_tokens replicated (see `lm_head_shard_follows_the_declared_size` for
    /// the packet-driven exception).
    #[test]
    fn lm_head_and_embed_are_replicated() {
        for n in [
            "lm_head.weight",
            "model.embed_tokens.weight",
            "model.layers.0.input_layernorm.weight",
            "model.norm.weight",
        ] {
            assert_eq!(shard_of(n), Shard::Replicated, "{n}");
        }
        for n in [
            "model.layers.3.self_attn.q_proj.weight",
            "model.layers.3.self_attn.k_proj.weight",
            "model.layers.3.self_attn.v_proj.weight",
            "model.layers.3.mlp.gate_proj.weight",
            "model.layers.3.mlp.up_proj.weight",
        ] {
            assert_eq!(shard_of(n), Shard::Column, "{n}");
        }
        for n in [
            "model.layers.3.self_attn.o_proj.weight",
            "model.layers.3.mlp.down_proj.weight",
        ] {
            assert_eq!(shard_of(n), Shard::Row, "{n}");
        }
    }

    /// Column shards partition the tensor: every rank gets a distinct, correctly
    /// offset piece, and the pieces reassemble into the original.
    #[test]
    fn column_shards_partition_the_matrix() {
        let full: Vec<u8> = (0..64u8).collect();
        let want = 16; // 4 shards
        let mut seen = Vec::new();
        for rank in 0..4 {
            let got =
                slice_for("q_proj.weight", &full, &[8, 8], want, rank, 4).unwrap();
            assert_eq!(&*got, &full[rank as usize * 16..(rank as usize + 1) * 16]);
            seen.extend_from_slice(&got);
        }
        assert_eq!(seen, full, "the shards must reassemble into the original");
    }

    /// THE kv-replication case, and the reason the index is not `rank`.
    ///
    /// Gemma-4's full layers have 4 KV heads; at tp=8 `k_proj` cannot split
    /// eight ways, so devgen declares a 4-way shard and ranks {0,1} share piece
    /// 0, {2,3} piece 1, and so on. Using `rank` directly would index piece 7 of
    /// a 4-piece tensor — off the end for half the ranks, and the wrong heads
    /// for the rest.
    #[test]
    fn fewer_shards_than_ranks_means_ranks_share_a_slice() {
        let full: Vec<u8> = (0..64u8).collect();
        let want = 16; // only 4 shards for 8 ranks
        for rank in 0..8u32 {
            let got = slice_for("k_proj.weight", &full, &[8, 8], want, rank, 8).unwrap();
            let piece = (rank / 2) as usize;
            assert_eq!(
                &*got,
                &full[piece * 16..(piece + 1) * 16],
                "rank {rank} must share piece {piece}"
            );
        }
    }

    /// Row shards take a strided COLUMN range: `in/N` elements out of every row.
    /// Getting this wrong by taking a contiguous range instead is the classic
    /// row-parallel bug and it produces a plausible-looking wrong tensor.
    #[test]
    fn row_shards_gather_a_column_range_from_every_row() {
        // [out=3, in=4] of u16 (bf16-width), value = row*10 + col.
        let mut full = Vec::new();
        for r in 0..3u16 {
            for c in 0..4u16 {
                full.extend_from_slice(&(r * 10 + c).to_le_bytes());
            }
        }
        // tp=2 => each rank owns 2 of the 4 columns, from all 3 rows.
        let want = 3 * 2 * 2;
        let r0 = slice_for("o_proj.weight", &full, &[3, 4], want, 0, 2).unwrap();
        let r1 = slice_for("down_proj.weight", &full, &[3, 4], want, 1, 2).unwrap();
        let u16s = |b: &[u8]| -> Vec<u16> {
            b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
        };
        assert_eq!(u16s(&r0), vec![0, 1, 10, 11, 20, 21]);
        assert_eq!(u16s(&r1), vec![2, 3, 12, 13, 22, 23]);
    }

    /// The same gather must serve fp8 (1 B/element) — the element width comes
    /// from the byte count, not from a precision flag that could disagree.
    #[test]
    fn row_shards_infer_the_element_width() {
        let full: Vec<u8> = (0..12u8).collect(); // [3,4] of 1-byte e4m3
        let got = slice_for("down_proj.weight", &full, &[3, 4], 6, 1, 2).unwrap();
        assert_eq!(&*got, &[2, 3, 6, 7, 10, 11]);
    }

    /// An fp8 per-output-channel SCALE is `[out]`: row-parallel splits the input
    /// axis, which it does not have, so it is replicated. Its twin WEIGHT is
    /// still sharded — the two travel together and disagree on purpose.
    #[test]
    fn row_parallel_scales_are_replicated_but_their_weights_are_not() {
        let scale: Vec<u8> = (0..16u8).collect(); // [4] of f32
        let got = slice_for("down_proj.weight_scale", &scale, &[4], 16, 1, 2).unwrap();
        assert_eq!(&*got, &scale[..], "a [out] scale has no input axis to cut");

        // Same layer, the weight itself, still gathered.
        let w: Vec<u8> = (0..12u8).collect();
        assert_eq!(
            &*slice_for("down_proj.weight", &w, &[3, 4], 6, 1, 2).unwrap(),
            &[2, 3, 6, 7, 10, 11]
        );
    }

    /// tp==1 must be byte-identical to the pre-TP path for every name, including
    /// the ones that WOULD be sharded.
    #[test]
    fn tp1_binds_everything_whole() {
        let full: Vec<u8> = (0..64u8).collect();
        for n in ["q_proj.weight", "o_proj.weight", "lm_head.weight"] {
            let got = slice_for(n, &full, &[8, 8], 64, 0, 1).unwrap();
            assert_eq!(&*got, &full[..], "{n}");
        }
    }

    /// Sizes that cannot be a shard are refused, not rounded — a silently wrong
    /// slice is a silently wrong token.
    #[test]
    fn impossible_shards_are_refused() {
        let full: Vec<u8> = (0..64u8).collect();
        // 64 is not a multiple of 20.
        assert!(slice_for("q_proj.weight", &full, &[8, 8], 20, 0, 4).is_err());
        // 8 shards do not divide tp=3.
        assert!(slice_for("q_proj.weight", &full, &[8, 8], 8, 0, 3).is_err());
        // Row-parallel with in=5 not divisible by tp=2.
        assert!(slice_for("o_proj.weight", &full, &[2, 5], 10, 0, 2).is_err());
        // Replicated but the sizes disagree.
        assert!(slice_for("lm_head.weight", &full, &[8, 8], 32, 0, 4).is_err());
    }
}

#[cfg(test)]
mod glm_shard_tests {
    use super::*;

    /// GLM-5.2 (MLA) has no q/k/v_proj — the absorbed `derived.*` weights carry the head axis, so
    /// they are what TP must split. Mirrors `glm_col()`/`glm_row()` in `glm52_decode.c`.
    #[test]
    fn glm_mla_derived_weights_are_column_parallel() {
        for n in [
            "model.layers.3.self_attn.derived.q_absorb.weight",
            "model.layers.3.self_attn.derived.q_rope.weight",
            "model.layers.3.self_attn.derived.v_absorb.weight",
        ] {
            assert_eq!(shard_of(n), Shard::Column, "{n}");
        }
    }

    /// The latent KV path is shared by every head: no head axis, so no split. Getting this wrong
    /// binds a quarter of the KV down-projection and the model silently degrades.
    #[test]
    fn glm_latent_kv_is_replicated() {
        for n in [
            "model.layers.3.self_attn.derived.kv_a_latent.weight",
            "model.layers.3.self_attn.derived.k_rope.weight",
        ] {
            assert_eq!(shard_of(n), Shard::Replicated, "{n}");
        }
    }

    /// A block-fp8 scale grid must shard on the same axis as the weight it scales — it rides the
    /// substring test rather than needing its own entry.
    #[test]
    fn glm_block_fp8_scales_follow_their_weight() {
        assert_eq!(
            shard_of("model.layers.3.mlp.shared_experts.gate_proj.weight_scale_inv"),
            Shard::Column
        );
        assert_eq!(
            shard_of("model.layers.3.mlp.shared_experts.down_proj.weight_scale_inv"),
            Shard::Row
        );
    }

    /// `GLM_LINEAR_FP8` emits `.weight_fp8` / `.weight_scale_inv` for `o_proj` and the three
    /// `shared_experts.*` projections, and it relies ENTIRELY on those names still containing the
    /// `<proj>.weight` substring these predicates match on. Nothing else states that dependency,
    /// and the failure mode is silent: a predicate tightened to an exact `ends_with(".weight")`
    /// would reclassify all eight names as `Replicated`, so every rank would bind the WHOLE tensor
    /// into a buffer the blob declared at 1/tp of its size. Pin both halves of every pair.
    #[test]
    fn glm_linear_fp8_names_shard_like_the_weights_they_replace() {
        for suffix in ["weight_fp8", "weight_scale_inv"] {
            assert_eq!(
                shard_of(&format!("model.layers.3.self_attn.o_proj.{suffix}")),
                Shard::Row,
                "o_proj is row-parallel (input lanes); {suffix}"
            );
            for proj in ["gate_proj", "up_proj"] {
                assert_eq!(
                    shard_of(&format!("model.layers.3.mlp.shared_experts.{proj}.{suffix}")),
                    Shard::Column,
                    "the shared expert's {proj} is column-parallel; {suffix}"
                );
            }
            assert_eq!(
                shard_of(&format!("model.layers.3.mlp.shared_experts.down_proj.{suffix}")),
                Shard::Row,
                "the shared expert's down_proj is row-parallel; {suffix}"
            );
        }
    }

    /// GLM's routed experts and o_proj already matched the pre-existing entries; pin that so a
    /// future edit to COL/ROW cannot quietly drop them.
    #[test]
    fn glm_expert_and_o_proj_classification_is_unchanged() {
        assert_eq!(
            shard_of("model.layers.3.mlp.experts.17.up_proj.weight"),
            Shard::Column
        );
        assert_eq!(
            shard_of("model.layers.3.mlp.experts.17.down_proj.weight"),
            Shard::Row
        );
        assert_eq!(shard_of("model.layers.3.self_attn.o_proj.weight"), Shard::Row);
        assert_eq!(shard_of("model.layers.3.mlp.gate.weight"), Shard::Replicated);
    }
}
