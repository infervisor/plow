//! Megatron weight sharding — which slice of a checkpoint tensor rank `r` owns.
//!
//! the design notes pairs a **column-parallel** producer with a
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
    //
    // The mxfp4 twin `down_proj.weight_scale` rides the SAME test for the same reason, but the
    // 1-D guard does NOT fire for it and must not: an E8M0 microscaling row is `[N, K/32]`, TWO
    // dimensions, so it is genuinely row-parallel and is gathered like its weight. (The 1-D guard
    // exists for the per-output-channel fp8 scale `[N]`, which is a different tensor that happens
    // to share the spelling. Which one a checkpoint has is decided by its SHAPE, here and in
    // `bind_packed_experts`, never by a flag.)
    const COL: [&str; 10] = [
        "q_proj.weight",
        "k_proj.weight",
        "v_proj.weight",
        "gate_proj.weight",
        "up_proj.weight",
        "derived.q_absorb",
        "derived.q_rope",
        "derived.v_absorb",
        // Mixtral-spelled routed experts -- see the note below.
        ".w1.weight",
        ".w3.weight",
    ];
    const ROW: [&str; 3] = ["o_proj.weight", "down_proj.weight", ".w2.weight"];

    // --- Mixtral `w1|w2|w3` routed experts (Kimi-K3), in the generic tables above ---
    //
    // K3's MoE block is `…block_sparse_moe.experts.{e}.w1|w2|w3` with mxfp4
    // `weight_packed` / `weight_scale` twins. w1=gate, w2=down, w3=up, and that
    // mapping is taken from the checkpoint's SHAPES rather than from the
    // convention's reputation: at latent=3584 / moe_inter=3072,
    //
    //     w1.weight_packed [3072, 1792] = [I_moe, latent/2]   latent -> I_moe
    //     w3.weight_packed [3072, 1792] = [I_moe, latent/2]   latent -> I_moe
    //     w2.weight_packed [3584, 1536] = [latent, I_moe/2]   I_moe  -> latent
    //
    // so w1/w3 carry the I_moe axis as their OUTPUT (column-parallel, like every
    // gate/up) and w2 carries it as its INPUT (row-parallel, like every down).
    //
    // `w1`/`w2`/`w3` match NEITHER generic entry on their own, so without these
    // three rows every routed-expert tensor on 92 layers falls through to
    // `Replicated`: no error and no missing weight, just a row-parallel `w2`
    // bound as a whole tensor into a buffer declared at 1/tp of its size — which
    // `slice_for` would then reject with a confusing message on the WEIGHT, and
    // accept silently on any tp where the sizes happened to line up.
    //
    // The entries are `.wN.weight`, not `.wN.weight_packed`, so the mxfp4 SCALE
    // twin `.wN.weight_scale` rides the same substring onto the same axis —
    // exactly as `down_proj.weight_scale_inv` rides `down_proj.weight`. The
    // leading dot is what stops them matching a longer identifier that merely
    // ENDS in `w1`.

    // --- Kimi-K3, tested BEFORE the generic tables ---
    //
    // Order is load-bearing: two K3 names CONTAIN a generic entry and would
    // otherwise be classified by accident, in opposite directions and both wrong.
    //
    //   routed_expert_up_proj.weight   contains "up_proj.weight"   -> Column
    //   routed_expert_down_proj.weight contains "down_proj.weight" -> Row
    //
    // `down` is [latent, hidden] and `up` is [hidden, latent].
    //
    // `down` STAYS REPLICATED and it is not an oversight. Its output feeds the routed
    // experts, which are sharded on their INTERMEDIATE and therefore each need the WHOLE
    // latent vector; so a sharded `down` — on either axis — needs a cross-rank gather or
    // reduce of its own, with no existing collective at that point in the layer to ride.
    // 92 more rendezvous per token buys back ~0.74 ms of streaming and costs at least a
    // packet each (~5.3 us x 92 = 0.49 ms), so it is close to a wash and is not taken.
    //
    // `up` IS SHARDED, column-parallel, and it reaches that answer through the generic
    // `up_proj.weight` entry below — which is why it is no longer listed here. Column is
    // correct for it: [hidden, latent] splits on `hidden`, its OUTPUT, so every element is
    // still a full-width dot product and the values are bit-identical to the replicated
    // emit. `emit_k3_latent_moe` folds the matching all-gather into the shared expert's
    // all-reduce, and `slice_for`'s `full == want` demotion keeps the tp=1 emit replicated.
    //
    // Getting `up`'s AXIS wrong is silent, which is why it is stated rather than left to
    // the substring: Column and Row both accept `want = full/tp`, so Row here would bind a
    // strided column gather where a contiguous row range was required and return a
    // same-sized, plausible, WRONG tensor — this module header's "classic row-parallel
    // bug". The `up_proj.weight` substring lands on Column, which is the right answer, and
    // `k3_latent_projections_shard_on_the_axis_the_emitter_declared` pins it.
    //
    // The KDA rows below mirror `devgen::kda::kda_shard_class`, which is the
    // reviewed specification (it panics on an unclassified name for exactly this
    // reason). It cannot be called from here: plowrt does not depend on devgen.
    // `kda_shard_class`'s two counter-intuitive entries are kept as it explains
    // them — `f_a_proj` is replicated because its output is the rank-128
    // bottleneck, and `o_norm` is [D], not [H*D].
    //
    // `b_proj.weight` deliberately also catches MLA's `q_b_proj` / `kv_b_proj`
    // (both end in it). Column is the correct answer for all three: each carries
    // the head axis. On a checkpoint whose emitter still declares them whole,
    // Column degenerates to the whole tensor (`n_shards = full/want = 1`), so
    // this is safe for the DeepSeek / Kimi-K2 MLA path, which GLM-5.2 avoids
    // entirely by prepping absorbed `derived.*` weights.
    const K3_REPLICATED: [&str; 3] = [
        "routed_expert_down_proj.weight",
        "f_a_proj.weight",
        "o_norm.weight",
    ];
    const K3_COL: [&str; 8] = [
        "g_proj.weight",
        "b_proj.weight",
        "q_conv1d.weight",
        "k_conv1d.weight",
        "v_conv1d.weight",
        "A_log",
        "dt_bias",
        "o_gate_proj.weight",
    ];
    if K3_REPLICATED.iter().any(|s| name.contains(s)) {
        return Shard::Replicated;
    }
    if K3_COL.iter().any(|s| name.contains(s)) {
        return Shard::Column;
    }

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
    let shard = if tp > 1 {
        shard_of(name)
    } else {
        Shard::Replicated
    };

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

    // KDA's `A_log` IS ZERO-PADDED IN THE CHECKPOINT, and it is the one tensor whose
    // stored length is not its logical one. It ships `[128]` = `head_dim` while only
    // `[:96]` = `num_heads` is live; `devgen::kda` states the contract
    // (`KDA_A_LOG_CKPT_LEN`) and `scripts/kda_verify_ckpt.py` verified it in 69/69
    // KDA layers — indices 0..95 non-zero, 96..127 exactly 0.0. SGLang's custom
    // `weight_loader` and vLLM PR #50089's `a_log_weight_loader` narrow the same way.
    //
    // It is per-HEAD, so it is genuinely column-parallel: at TP8 rank r owns heads
    // [12r, 12r+12) and must read exactly those f32s. But the generic Column arm
    // recovers `n_shards` as `full / want`, and 512 / 48 is not an integer — so the
    // load failed on byte size at EVERY tp, including tp=1 (512 stored vs 384
    // declared). That failure is the good outcome the padding note predicted: the
    // alternative is consuming all 128 as if per-head-dim, which computes the wrong
    // decay for every token of every KDA layer and looks entirely fluent.
    //
    // The live prefix is `want * tp` and the tail is padding, so the rank's slice is
    // the ordinary contiguous one taken from the FRONT. Head-major and contiguous is
    // what the TP1 numeric gates already exercise — they read 0..95 in head order and
    // match the reference at 1.6e-04 on the decay stage — so this is the same
    // indexing one rank at a time, not a new assumption.
    //
    // Keyed on the name because the padding is a property of THIS tensor, not a
    // general licence to ignore a size mismatch: every other Column tensor still has
    // to divide exactly.
    //
    // AND ON `shard_of` RATHER THAN ON `shard`, which is the whole reason single-GPU
    // K3 could not load. `shard` was forced to `Replicated` at the top of this
    // function whenever `tp == 1`, so a test for `Shard::Column` here was
    // UNREACHABLE at tp = 1 and every one-GPU K3 bind died on
    //
    //     shard …self_attn.A_log: replicated but the checkpoint has 512 B and the
    //     blob declares 384 B
    //
    // — 69 times, at load, before a single token. The padding is a fact about the
    // TENSOR (stored [128] = head_dim, live [96] = num_heads), not about the
    // degree, and `shard_of` is where facts about a tensor live. At tp = 1 the arm
    // takes `rank * want = 0` and returns the live prefix, which is exactly what the
    // TP1 numeric gates read.
    //
    // It matters more than a one-GPU convenience: with no tp = 1 path there was no
    // way to A/B the 12-local-head, BV=8 TP8 KDA geometry against the 96-head, BV=16
    // one the block gates actually validate.
    if tp > 1 || full != want {
        if name.ends_with("A_log") && shard_of(name) == Shard::Column {
            let live = want.checked_mul(tp as u64).unwrap_or(0);
            if want != 0 && live <= full {
                let off = (rank as u64 * want) as usize;
                return Ok(Cow::Borrowed(&src[off..off + want as usize]));
            }
        }
    }

    // THE BLOB DECIDES WHETHER THIS RANK WANTS A SLICE. Same principle as the
    // `lm_head` rule above, generalised: a name tells you which AXIS a tensor would
    // shard on, and nothing else. Whether the emitter CHOSE to shard it is a fact
    // about the emitter, and the emitter already stated it — by declaring a buffer
    // of `full` bytes instead of `full / n`.
    //
    // KIMI-K3 IS WHY. `shard_of` classifies by substring, so K3's LatentMoE lands
    // four tensors per layer in the wrong bucket:
    //
    //     mlp.routed_expert_down_proj.weight   contains "down_proj.weight" -> Row
    //     mlp.routed_expert_up_proj.weight     contains "up_proj.weight"   -> Column
    //     mlp.shared_experts.{gate,up}_proj    -> Column      .down_proj   -> Row
    //
    // and `emit_k3_latent_moe` replicates every one of them. That is a DELIBERATE
    // choice, not an oversight: it shards the expert intermediate (`imoe =
    // tp.local(moe_inter)`, the 896-expert axis that actually dominates) and keeps
    // the two latent projections whole because `routed_expert_norm` is nonlinear —
    // normalising a partial sum would be finite, plausible and wrong — and keeps
    // the shared expert whole because sharding it would need a THIRD all-reduce per
    // layer, 92 more collectives per token, against a 2048-wide intermediate. At
    // batch-1 decode that trade is not close.
    //
    // The obvious alternative — teach `shard_of` that `shared_experts.*` is
    // replicated — is WRONG and would have broken a shipping model:
    // `glm_linear_fp8_names_shard_like_the_weights_they_replace` pins GLM-5.2's
    // shared expert as genuinely Column/Row, and `mla.rs` sizes it at `imoe_l =
    // imoe / tp`. The same spelling is sharded in one model and replicated in
    // another, so no name table can answer this. The declared size can.
    //
    // Narrow on purpose: `full == want` EXACTLY. A tensor the emitter meant to
    // shard declares `full / n`, never `full`, so this cannot swallow a genuine
    // emitter/loader disagreement — that still lands in the arms below and still
    // errors. And because the emitter derives the declared bytes and the kernel's
    // width from the same `tp.local()` call, "declared whole" and "computed whole"
    // cannot come apart.
    let shard = match shard {
        Shard::Column | Shard::Row if want != 0 && full == want => Shard::Replicated,
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
                return Err(bad(format!(
                    "row-parallel with a degenerate shape {shape:?}"
                )));
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

    /// KIMI-K3's LATENT MoE REPLICATES FOUR TENSORS PER LAYER THAT `shard_of` NAMES
    /// AS SHARDED, AND THE BLOB IS WHAT SAYS SO.
    ///
    /// Before this rule a K3 packet could not be LOADED at any tp > 1: the emitter
    /// declares these whole, `shard_of` matched `down_proj.weight` /
    /// `up_proj.weight` / `gate_proj.weight` inside the longer K3 names, and the
    /// Row arm demanded a 1/tp slice of a buffer sized for the whole thing. The
    /// load died before a single expert bound.
    #[test]
    fn k3_replicates_the_latent_and_shared_projections_at_every_tp() {
        const OUT: usize = 8;
        const IN: usize = 8;
        let full: Vec<u8> = (0..OUT * IN * 2).map(|i| (i / 8) as u8).collect();
        let whole = full.len() as u64;
        let p = "language_model.model.layers.5.mlp.";
        for n in [
            "routed_expert_down_proj.weight",
            "routed_expert_up_proj.weight",
            "shared_experts.gate_proj.weight",
            "shared_experts.up_proj.weight",
            "shared_experts.down_proj.weight",
        ] {
            let name = format!("{p}{n}");
            // Every rank of a TP8 group binds the identical whole tensor.
            for rank in 0..8 {
                let got = slice_for(&name, &full, &[OUT, IN], whole, rank, 8)
                    .unwrap_or_else(|e| panic!("{name} at rank {rank}: {e}"));
                assert_eq!(&*got, &full[..], "{name} rank {rank} did not bind whole");
            }
        }
    }

    /// THE NEGATIVE CONTROL, and it is the whole reason the rule is `full == want`
    /// exactly rather than "if the Row arm would fail, replicate".
    ///
    /// A tensor the emitter MEANT to shard declares `full / n`, never `full`. Those
    /// still take the sharding arms, and a declared size that matches neither still
    /// errors — so a genuine emitter/loader disagreement is not swallowed by the
    /// replicated escape hatch.
    #[test]
    fn a_declared_slice_still_shards_and_a_wrong_one_still_errors() {
        let full: Vec<u8> = (0..64).map(|i| i as u8).collect();
        // Declared at 1/4: genuinely column-parallel, rank 2 gets the third quarter.
        let got = slice_for("gate_proj.weight", &full, &[8, 4], 16, 2, 4).unwrap();
        assert_eq!(
            &*got,
            &full[32..48],
            "a declared 1/tp slice stopped sharding"
        );
        // Row-parallel declared at 1/2 still gathers rather than binding whole.
        let g2 = slice_for("down_proj.weight", &full, &[8, 4], 32, 1, 2).unwrap();
        assert_eq!(g2.len(), 32);
        assert_ne!(
            &*g2,
            &full[..32],
            "row-parallel returned a contiguous prefix"
        );
        // Neither whole nor a clean 1/tp: still an error, not a silent replicate.
        assert!(slice_for("gate_proj.weight", &full, &[8, 4], 20, 0, 4).is_err());
    }

    /// GLM-5.2 SHARDS THE SAME SPELLING K3 REPLICATES, WHICH IS WHY NO NAME TABLE
    /// CAN ANSWER THIS. `mla.rs` sizes GLM's shared expert at `imoe_l = imoe / tp`,
    /// so its blob declares 1/tp and it must keep sharding.
    #[test]
    fn glms_shared_expert_still_shards_because_its_blob_asks_for_a_slice() {
        let full: Vec<u8> = (0..64).map(|i| i as u8).collect();
        let n = "model.layers.3.mlp.shared_experts.gate_proj.weight";
        assert_eq!(shard_of(n), Shard::Column);
        let got = slice_for(n, &full, &[8, 4], 16, 1, 4).unwrap();
        assert_eq!(&*got, &full[16..32], "GLM's shared expert stopped sharding");
    }

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
            let got = slice_for("q_proj.weight", &full, &[8, 8], want, rank, 4).unwrap();
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
            b.chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect()
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

    /// KDA's zero-padded `A_log` narrows at EVERY degree, `tp == 1` included.
    ///
    /// The checkpoint stores `[head_dim] = 128` f32 and only `[:num_heads] = 96` is
    /// live, so the declared bytes are always smaller than the stored ones and the
    /// generic Replicated arm's `full == want` check can never pass. The narrow used
    /// to be keyed on the ALREADY-TP-COLLAPSED `shard`, which `slice_for` forces to
    /// `Replicated` whenever `tp == 1` — so it was unreachable on one GPU and all 69
    /// KDA layers failed at bind with "replicated but the checkpoint has 512 B and
    /// the blob declares 384 B". No single-GPU K3 could load, which also meant no
    /// tp = 1 baseline existed to A/B the TP8 KDA geometry against.
    #[test]
    fn the_padded_a_log_narrows_at_every_degree_including_tp1() {
        const N: &str = "language_model.model.layers.2.self_attn.A_log";
        // 128 f32 stored, 96 live, head-major.
        let full: Vec<u8> = (0..128u32).flat_map(|i| (i as f32).to_le_bytes()).collect();
        let f32s = |b: &[u8]| -> Vec<f32> {
            b.chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect()
        };

        // tp = 1: the whole live prefix, and NOT the 32 pad entries.
        let got = slice_for(N, &full, &[128], 96 * 4, 0, 1).unwrap();
        assert_eq!(got.len(), 96 * 4);
        assert_eq!(f32s(&got), (0..96).map(|i| i as f32).collect::<Vec<_>>());

        // tp = 8: rank r owns heads [12r, 12r + 12), contiguous from the FRONT.
        for r in 0..8u32 {
            let got = slice_for(N, &full, &[128], 12 * 4, r, 8).unwrap();
            assert_eq!(
                f32s(&got),
                (0..12).map(|i| (12 * r + i) as f32).collect::<Vec<_>>(),
                "rank {r}"
            );
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
                    shard_of(&format!(
                        "model.layers.3.mlp.shared_experts.{proj}.{suffix}"
                    )),
                    Shard::Column,
                    "the shared expert's {proj} is column-parallel; {suffix}"
                );
            }
            assert_eq!(
                shard_of(&format!(
                    "model.layers.3.mlp.shared_experts.down_proj.{suffix}"
                )),
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
        assert_eq!(
            shard_of("model.layers.3.self_attn.o_proj.weight"),
            Shard::Row
        );
        assert_eq!(
            shard_of("model.layers.3.mlp.gate.weight"),
            Shard::Replicated
        );
    }

    /// K3's two latent-MoE projections shard on DIFFERENT rules, and both are pinned
    /// because both failures are silent.
    ///
    /// `up` is [hidden, latent] and is COLUMN-parallel: it splits its OUTPUT, which is
    /// what makes the shard bit-neutral (every element is still a full-latent dot
    /// product) and what lets `emit_k3_latent_moe` fold the all-gather into the shared
    /// expert's all-reduce. It reaches Column through the generic `up_proj.weight`
    /// entry — and Row would ALSO accept `want = full/tp`, binding a strided column
    /// gather where a contiguous row range was required and returning the wrong bytes
    /// with no error.
    ///
    /// `down` is [latent, hidden] and stays REPLICATED: its output feeds experts that
    /// are sharded on their intermediate and each need the whole latent, so any shard
    /// of it needs a collective of its own. Left to the substring it would be Row.
    #[test]
    fn k3_latent_projections_shard_on_the_axis_the_emitter_declared() {
        const P: &str = "language_model.model.layers.5.mlp.";
        assert_eq!(
            shard_of(&format!("{P}routed_expert_up_proj.weight")),
            Shard::Column,
            "routed_expert_up_proj is column-parallel: [hidden, latent] splits on hidden"
        );
        assert_eq!(
            shard_of(&format!("{P}routed_expert_down_proj.weight")),
            Shard::Replicated,
            "routed_expert_down_proj inherited Row from `down_proj.weight`"
        );
        assert_eq!(
            shard_of(&format!("{P}routed_expert_norm.weight")),
            Shard::Replicated
        );
    }

    /// The column shard `up` actually gets: rank r's contiguous row range of
    /// [hidden, latent], and the eight of them reassemble into the checkpoint tensor.
    ///
    /// A Row classification would pass every size check here and return `full[2],
    /// full[3], full[6], ...` instead — same length, same dtype, wrong model. That is
    /// the reason this asserts BYTES and not just a length.
    #[test]
    fn k3_up_proj_column_shard_is_a_contiguous_row_range() {
        const H: usize = 8; // "hidden": the split axis
        const L: usize = 4; // "latent"
        let full: Vec<u8> = (0..(H * L * 2) as u8).collect();
        let want = (H * L * 2 / 4) as u64;
        let name = "language_model.model.layers.5.mlp.routed_expert_up_proj.weight";
        let mut seen = Vec::new();
        for rank in 0..4 {
            let got = slice_for(name, &full, &[H, L], want, rank, 4).unwrap();
            assert_eq!(
                &*got,
                &full[rank as usize * want as usize..(rank as usize + 1) * want as usize],
                "rank {rank} must own hidden rows [{}, {})",
                rank as usize * H / 4,
                (rank as usize + 1) * H / 4
            );
            seen.extend_from_slice(&got);
        }
        assert_eq!(
            seen, full,
            "the eight shards must reassemble into the original"
        );
        // And declared WHOLE it is still replicated — the tp=1 emit, unchanged.
        let whole = full.len() as u64;
        assert_eq!(
            &*slice_for(name, &full, &[H, L], whole, 3, 4).unwrap(),
            &full[..]
        );
    }

    /// Every KDA tensor, against `devgen::kda::kda_shard_class` — the reviewed
    /// specification this table mirrors. plowrt cannot call it (no devgen
    /// dependency), so the agreement is pinned here instead. Seven of these were
    /// silently Replicated before, which for KDA is "not a crash, just wrong math
    /// on >1 GPU".
    #[test]
    fn kda_tensors_match_the_devgen_shard_specification() {
        const P: &str = "language_model.model.layers.2.self_attn.";
        for n in [
            "q_proj.weight",
            "k_proj.weight",
            "v_proj.weight",
            "g_proj.weight",
            "b_proj.weight",
            "f_b_proj.weight",
            "q_conv1d.weight",
            "k_conv1d.weight",
            "v_conv1d.weight",
            "A_log",
            "dt_bias",
        ] {
            assert_eq!(
                shard_of(&format!("{P}{n}")),
                Shard::Column,
                "KDA `{n}` must be column-parallel"
            );
        }
        assert_eq!(shard_of(&format!("{P}o_proj.weight")), Shard::Row);
        for n in ["f_a_proj.weight", "o_norm.weight"] {
            assert_eq!(
                shard_of(&format!("{P}{n}")),
                Shard::Replicated,
                "KDA `{n}` must be replicated"
            );
        }
    }

    /// K3's MLA half. `o_gate_proj` was already Column, but only because it
    /// CONTAINS `gate_proj.weight`; pin it so the accident cannot be undone by an
    /// edit to the generic table. `q_b_proj`/`kv_b_proj` carry the head axis and
    /// are caught by the `b_proj.weight` entry.
    #[test]
    fn k3_mla_projections_split_on_the_head_axis() {
        const P: &str = "language_model.model.layers.3.self_attn.";
        assert_eq!(shard_of(&format!("{P}o_gate_proj.weight")), Shard::Column);
        assert_eq!(shard_of(&format!("{P}q_b_proj.weight")), Shard::Column);
        assert_eq!(shard_of(&format!("{P}kv_b_proj.weight")), Shard::Column);
        // The shared latent path has no head axis to cut.
        assert_eq!(shard_of(&format!("{P}q_a_proj.weight")), Shard::Replicated);
        assert_eq!(
            shard_of(&format!("{P}kv_a_proj_with_mqa.weight")),
            Shard::Replicated
        );
        assert_eq!(shard_of(&format!("{P}o_proj.weight")), Shard::Row);
    }
}

#[cfg(test)]
mod mxfp4_shard_tests {
    use super::*;

    const K3: &str = "language_model.model.layers.5.block_sparse_moe.experts.17.";

    /// K3's routed experts are spelled `w1|w2|w3`, which matches no generic
    /// entry, and the default of this classifier is `Replicated`. Both halves of
    /// each mxfp4 pair — `weight_packed` and its `weight_scale` twin — must land
    /// on the axis the projection actually has.
    #[test]
    fn mixtral_experts_shard_on_the_axis_their_shapes_carry() {
        for suffix in ["weight_packed", "weight_scale"] {
            for gate_or_up in ["w1", "w3"] {
                assert_eq!(
                    shard_of(&format!("{K3}{gate_or_up}.{suffix}")),
                    Shard::Column,
                    "{gate_or_up}.{suffix} is [I_moe, latent/2] — output-parallel"
                );
            }
            assert_eq!(
                shard_of(&format!("{K3}w2.{suffix}")),
                Shard::Row,
                "w2.{suffix} is [latent, I_moe/2] — input-parallel"
            );
        }
    }

    /// The leading dot is load-bearing: a longer identifier that merely ENDS in
    /// `w1` must not be dragged into the expert tables.
    #[test]
    fn the_mixtral_entries_do_not_capture_longer_identifiers() {
        for n in [
            "model.layers.0.self_attn.qw1.weight",
            "model.layers.0.mlp.rw2.weight",
        ] {
            assert_eq!(shard_of(n), Shard::Replicated, "{n}");
        }
    }

    /// An mxfp4 `weight_scale` is `[N, K/32]` — TWO dimensions — so the 1-D
    /// demotion must NOT fire and the row shard must be a real column gather at
    /// group granularity. The per-output-channel fp8 scale `[N]` shares the
    /// spelling and IS demoted; the shape is what tells them apart, and getting
    /// it wrong is silent because both produce the right NUMBER of bytes.
    #[test]
    fn a_two_dimensional_e8m0_scale_row_is_gathered_not_replicated() {
        // [out=3, K/32=4] of E8M0, value = row*10 + group.
        let full: Vec<u8> = (0..3u8)
            .flat_map(|r| (0..4u8).map(move |g| r * 10 + g))
            .collect();
        let name = format!("{K3}w2.weight_scale");
        let got = slice_for(&name, &full, &[3, 4], 6, 1, 2).unwrap();
        assert_eq!(
            &*got,
            &[2, 3, 12, 13, 22, 23],
            "rank 1 owns the upper K groups"
        );

        // Same name, the per-channel fp8 shape: no input axis, so replicated.
        let ch: Vec<u8> = (0..12u8).collect();
        let got = slice_for(
            "model.layers.3.mlp.down_proj.weight_scale",
            &ch,
            &[3],
            12,
            1,
            2,
        )
        .unwrap();
        assert_eq!(&*got, &ch[..]);
    }

    /// The mxfp4 twins must agree with each other, which they only do if the
    /// PACKED weight and its scale are cut at the same K boundary. `w2` is
    /// [latent, I_moe/2] packed and [latent, I_moe/32] scaled, so a tp-way cut
    /// takes I_moe/(2*tp) payload bytes and I_moe/(32*tp) scale bytes per row.
    #[test]
    fn the_packed_weight_and_its_scale_are_cut_at_the_same_k_boundary() {
        const LAT: usize = 8; // rows
        const IMOE: usize = 128; // K
        const TP: u32 = 2;
        let w: Vec<u8> = (0..LAT * IMOE / 2).map(|i| i as u8).collect();
        let s: Vec<u8> = (0..LAT * IMOE / 32).map(|i| i as u8).collect();
        let wn = format!("{K3}w2.weight_packed");
        let sn = format!("{K3}w2.weight_scale");
        for rank in 0..TP {
            let gw = slice_for(
                &wn,
                &w,
                &[LAT, IMOE / 2],
                (w.len() / TP as usize) as u64,
                rank,
                TP,
            )
            .unwrap();
            let gs = slice_for(
                &sn,
                &s,
                &[LAT, IMOE / 32],
                (s.len() / TP as usize) as u64,
                rank,
                TP,
            )
            .unwrap();
            // Row 0's first payload byte holds elements [rank*IMOE/TP ..], and its
            // first scale byte covers exactly that group.
            assert_eq!(gw[0], (rank as usize * IMOE / 2 / TP as usize) as u8);
            assert_eq!(gs[0], (rank as usize * IMOE / 32 / TP as usize) as u8);
            // 2 fp4 per payload byte, 32 elements per scale byte.
            assert_eq!(gw.len() * 2, gs.len() * 32);
        }
    }
}
