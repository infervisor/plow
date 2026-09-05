//! KV-cache write-row sites and prefill-chunk rebasing — the ONE rule for
//! "which instruction field carries a KV row", shared by every engine.
//!
//! Pure functions over the device ISA (`DevInst64`, `DevOp`, tensor names), no
//! vendor types: `exec::amd` and `exec::cpu` both patch programs through here so
//! decode and prefill cannot drift apart again (see `derive_kvrow`).

use packet::dev::{DevInst64, DevOp};

use crate::asset::devblob::DevProg;

pub(crate) fn kvrow_span(kvrow: &[u32]) -> Option<(usize, usize)> {
    let lo = *kvrow.iter().min()? as usize;
    let hi = *kvrow.iter().max()? as usize;
    Some((lo, hi))
}

/// KV-append sites for a packet that DECLARED NONE — returned as
/// `(i[3] sites, i[2] sites)`.
///
/// [`DevBlob::kvrow`] is a list of instructions whose `i[3]` is the write row:
/// one field, one op family, which is everything a GQA decode needs. GLM-5.2's
/// MLA does not fit it. Its cache is a latent + a rope half written by TWO
/// different ops with the row in DIFFERENT fields — the `RmsNorm` into
/// `kv.L.ckv` carries it in `i[2]`, the `HeadNormRope` into `kv.L.krot` in
/// `i[3]` — so `devgen` declares none of them at all (`n_kvrow = 0`).
///
/// A host that then patches nothing does not fail: every token's KV lands in
/// row 0, attention reads `[0, kvlen)` of a cache that never advanced, and the
/// model emits fluent-looking ids that are wrong from the second token on. That
/// is why this is derived rather than skipped. `runtime/tests/glm52_decode.c`
/// finds the same sites by the same rule — the op, plus a `kv.` destination.
///
/// Only consulted when the packet declared no sites, so a Gemma packet keeps
/// the compiled list untouched.
pub(crate) fn derive_kvrow(p: &DevProg, names: &[String]) -> (Vec<u32>, Vec<u32>) {
    let (mut i3, mut i2) = (Vec::new(), Vec::new());
    for (k, d) in p.insts.iter().enumerate() {
        match kv_write_row_field(d.op, names.get(d.t[0] as usize)) {
            Some(3) => i3.push(k as u32),
            Some(2) => i2.push(k as u32),
            _ => {}
        }
    }
    (i3, i2)
}

/// Which `i[]` field of an instruction carries the KV-cache WRITE ROW, or `None`
/// when the instruction does not write the cache at all.
///
/// ONE rule, consulted by BOTH phases — [`derive_kvrow`] for the decode step's
/// per-token row and [`rebase_chunk`] for a prefill chunk's base row. They used
/// to disagree: decode found the sites by destination NAME, prefill by
/// `HeadNormRope` + `fj[1] != 0`, and the second test matches nothing on GLM's
/// MLA (its k_rope sets `j[1]`, which packs into `fj[2]`, and leaves `f[1]`/`j[0]`
/// — the two halves of `fj[1]` — at zero). So a GLM prefill chunk wrote every
/// latent row of every chunk at row 0 and only the LAST chunk's tail survived,
/// with no error anywhere. Two phases deriving "what is a KV write" by two rules
/// is the bug; this is the one rule.
///
/// The destination NAME is the discriminator, and it has to be: the field alone
/// cannot tell GLM's `kv_a_layernorm` (an `RmsNorm` whose `i[2]` is the latent
/// out-row) from the input/post-attention norms, which are the same opcode with
/// `i[2]` meaning nothing. Keying on the opcode alone would rebase those two and
/// corrupt the block input.
///
/// The `HeadNormRopeFp8` twin is included for the same reason the flash test
/// includes its own: an fp8-KV packet emits the fp8 opcode, and a bf16-only test
/// silently matches nothing there — the class of miss `derive_segments` records.
pub(crate) fn kv_write_row_field(op: u16, dst: Option<&String>) -> Option<usize> {
    if !dst.is_some_and(|n| n.starts_with("kv.")) {
        return None;
    }
    if op == DevOp::RmsNorm as u16 {
        // GLM/DeepSeek MLA: `kv_a_layernorm` -> `kv.{L}.ckv`, out_row0 in i[2]
        // (`runtime/amd/interp.hip`, PLOW_DOP_RMSNORM).
        Some(2)
    } else if op == DevOp::DsaPoolCompress as u16 {
        // GLM-5.3 pooled indexer: prefill pool-cache base in i[3].
        Some(3)
    } else if op == DevOp::HeadNormRope as u16 || op == DevOp::HeadNormRopeFp8 as u16 {
        // Dense GQA k/v norm -> `kv.{L}.k`/`.v`, and MLA's k_rope -> `kv.{L}.krot`.
        // Both carry the write row in i[3].
        Some(3)
    } else {
        None
    }
}

/// Rebase one prefill program's instructions onto the chunk `[c0, c0+clen)`.
///
/// Split out of [`AmdEngine::patch_prefill`] so the rule can be tested without a
/// GPU: every one of these families was, at some point, patched positionally and
/// silently wrong, and a positional bug is invisible to anything but a test that
/// inspects the fields.
///
/// FOUR patch families, every one found BY IDENTITY rather than by position:
///
/// * a **KV-write site** ([`kv_write_row_field`]) → its row field = `c0`. This
///   covers dense GQA's k/v norm (`i[3]`), MLA's k_rope (`i[3]`) and MLA's latent
///   `kv_a_layernorm` (`i[2]`) with one rule, so prefill and decode cannot drift
///   about what a KV write is.
/// * `HeadNormRope`/`HeadNormRopeFp8` **with `fj[1] != 0`** → `i[3] = c0`. The
///   legacy dense-GQA test, kept because it keys on the KV RING STRIDE (`j[0]`,
///   which packs into `fj[1]`) rather than on a tensor name: a packet whose KV
///   tensors are not named `kv.*` still rebases. Its `fj[1] == 0` twin is the
///   *query* norm and must be left alone; patching it corrupts Q with a cache
///   row index. The two tests are a UNION, and on Gemma they agree site for site.
/// * `FlashPrefill`/`FlashPrefillFp8` → `i[4] = c0` (q_pos0) and
///   `i[1] = c0 + clen` (n_kv, everything written so far, not just this chunk).
/// * every **KDA op** ([`KDA_ROW_COUNT_OPS`]) → `i[0] = clen`, the REAL row count.
/// * `DsaPoolCompress` (GLM-5.3's pooled indexer, prefill mode only — this
///   function never sees a decode program) → `i[0] = clen / pool_size`, not the
///   baked `t / pool_size`. Same reasoning as KDA: the pool cache is PERSISTENT
///   (`kv.*`, addressed by [`kv_write_row_field`]'s `chunk_base` above), so a
///   padded chunk's pool count must stop at the last pool `clen` actually
///   completes — one pool further and the boundary pool mixes real tokens with
///   pad-row garbage, silently corrupting a cache entry decode will later select.
///   `clen == t` (the common, non-ragged case) reproduces the baked value exactly.
/// * `DsaPoolStash` (its prefill TAIL-SEED twin) → carries the `clen % pool_size`
///   real trailing rows `DsaPoolCompress`'s shrink above deliberately leaves
///   uncompressed into the decode-side ring, so that boundary pool completes
///   correctly once decode contributes the rest instead of starting from
///   whatever the ring happened to hold. `devgen` bakes `pool_size - 1` of these
///   at `row = t - 1 - k`; rebased here to `row = clen - 1 - k` (an offset from
///   `clen`'s end instead of `t`'s) and disabled (`Nop`) once `row` would fall
///   inside a pool `DsaPoolCompress` already compressed — a live "genuine" and
///   "surplus" write can share a ring slot (`pos % pool_size` cycles every
///   `pool_size` rows), and letting a stale already-cached row overwrite a real
///   one is exactly the bug this rebase exists to prevent, not a rarer version
///   of it. On a full chunk (`clen == t`) `complete == clen`, so every baked row
///   is `< complete` by construction and all of them `Nop` — correctly: every
///   pool in a full chunk, including its last, was already compressed directly
///   by `DsaPoolCompress` itself, so there is nothing left for the ring to carry.
///
/// # Why KDA needs a row count where attention needs only a bound
///
/// The flash family above leaves its row count alone and bounds the KV *reads* at
/// `c0 + clen`: a padded row computes a garbage output that nothing ever reads,
/// because the lm_head samples row `clen - 1`. That convention is safe for
/// attention precisely because attention is STATELESS across the token axis —
/// each row's output depends only on rows behind it, so a junk row poisons only
/// itself.
///
/// KDA is not. Both of its arms CARRY STATE FORWARD along `t`, and the loop bound
/// is the baked `T`:
///
/// * `op_kda.h` conv — `for (t = 0; t < T; t++)` rolls the window left and shifts
///   `x[t]` into the newest tap. A zero pad row is not ignored; it is CONVOLVED,
///   and it evicts a real tap from a `W`-wide window. After `T - clen` pad rows a
///   `W = 4` window holds nothing but zeros.
/// * `op_kda.h` recurrence — the same `for (t = 0; t < T; t++)`, and each step
///   applies the decay `exp(a_log[h])` to the carried state. Pad rows contribute
///   no `k^T v` outer product, but they DO decay: the state handed to decode has
///   been multiplied by an extra `exp(a_log)^(T - clen)`.
///
/// So the state left after a chunk belongs to the padded BUCKET WIDTH rather than
/// to the prompt. Nothing reads the pad rows' outputs, but the next chunk and the
/// whole decode phase read the STATE, and it is wrong for every prompt that is not
/// exactly a bucket multiple — i.e. almost all of them. Setting `i[0] = clen` is
/// what makes a K3 prefill stop at the last real token.
///
/// `FlashMlaPrefill`/`FlashGatherPrefill` are deliberately NOT here. Their query
/// base is not an immediate: `d_flash_mla_decode` (the body both prefill wrappers
/// call) derives `qpos = kv_len[b] - n_tok + t` from the `in.kvlen` TENSOR, with
/// `n_tok` in `i[4]` and the causal end clamped to `qpos + 1`. So the chunk base
/// arrives through the `in.kvlen` upload in [`AmdEngine::prefill_prepare`], and
/// writing `c0` into any of `i[0..7]` here would overwrite a live operand
/// (n_batch / n_head / kv_stride / window / n_tok / kv_mask / grouping).
///
/// The fp8 twins are included in the flash test. Their absence is not a slowdown,
/// it is silence: on an fp8-KV packet the bf16-only test matches nothing, so every
/// flash window stays at whatever the compiler baked in. (Same root cause as the
/// class-4 miss in [`derive_segments`], which found 0 of 60 flash segments on such
/// a packet.) Note the two are INDEPENDENT axes: an fp8-*weight* packet keeps a
/// bf16 KV and emits plain `FlashPrefill`, so this must key on the OPCODE and
/// never on a precision flag.
/// Every KDA opcode whose `i[0]` is the token-row count `T`.
///
/// Every mixer stage must see one uniform `clen`: legacy and chunk recurrences
/// carry state, while their surrounding stages consume the shortened rows.
pub(crate) const KDA_ROW_COUNT_OPS: &[DevOp] = &[
    DevOp::KdaConv,
    DevOp::KdaConv3,
    DevOp::KdaGate,
    DevOp::KdaStateStep,
    DevOp::KdaStateStepG,
    DevOp::KdaConvStateStepG,
    DevOp::KdaGatedNorm,
    DevOp::KdaChunkPrepare,
    DevOp::KdaChunkIntra,
    DevOp::KdaChunkWu,
    DevOp::KdaChunkCarry,
];

/// Where a prefill op carries its TOKEN-ROW COUNT, and whether that field is the
/// row count itself or a whole multiple of it.
///
/// Read by [`rebase_chunk_rows`] under `PLOW_RAGGED_CHUNK`; see its header for
/// why the multiple form exists and what the "== bucket width" guard buys.
///
/// `Rows` = the field IS `T`. `RowsTimes` = the field is `T * F` for some
/// per-instruction `F` (an element count over `[T, F]`), so it is RESCALED
/// rather than overwritten.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum RowField {
    Rows(usize),
    RowsTimes(usize),
}

/// The row-count field of every opcode a GLM/MLA prefill bucket emits.
///
/// DERIVED FROM THE PACKET, NOT FROM MEMORY: the op list is exactly the census of
/// `plowrt disasm --program 4096` on the shipped GLM-5.2 TP8 blob, and each field
/// index is the one `runtime/common/dev_isa.h` documents for that opcode. Four
/// opcodes in that census are deliberately ABSENT:
///
/// * `MoeGroupGluPf` / `MoeGroupDownPf` — they carry no `T` at all. Their work is
///   the sorted-row `meta` table `MoeAlignPf` builds, so shrinking align's `T`
///   shrinks them with no field of their own to patch. (This is also why a SHORT
///   chunk costs far less MoE than a padded one: with `clen` rows only
///   `clen * top_k` expert slots exist, so most of the 256 experts contribute no
///   tile and their weights are never streamed.)
/// * `Gemv` — the lm_head, `M = 1` with `a_row0` placed by
///   [`AmdEngine::patch_prefill`]. Its M is not a row count.
/// * `Argmax` / `XArgmaxFin` — vocabulary-dimensioned.
///
/// `FlashPrefill`/`FlashPrefillFp8` (dense GQA) are absent because MLA is the only
/// prefill family this axis has been measured on; leaving them out means a
/// non-MLA model under the flag runs the padded bucket — correct-just-slower
/// rather than untested.
pub(crate) const PREFILL_ROW_FIELDS: &[(DevOp, RowField)] = &[
    (DevOp::Embed, RowField::Rows(0)),
    (DevOp::RmsNorm, RowField::Rows(0)),
    (DevOp::HeadNormRope, RowField::Rows(0)),
    (DevOp::HeadNormRopeFp8, RowField::Rows(0)),
    (DevOp::Residual, RowField::RowsTimes(0)),
    (DevOp::Glu, RowField::RowsTimes(0)),
    (DevOp::Gemm, RowField::Rows(0)),
    (DevOp::GemmNorm, RowField::Rows(0)),
    (DevOp::GemmSmall, RowField::Rows(0)),
    (DevOp::GemmMed, RowField::Rows(0)),
    (DevOp::GemmWide, RowField::Rows(0)),
    (DevOp::GemmC5, RowField::Rows(0)),
    (DevOp::GemmGlu, RowField::Rows(0)),
    (DevOp::GemmFp8, RowField::Rows(0)),
    (DevOp::GemmMedFp8, RowField::Rows(0)),
    (DevOp::GemmSmallFp8, RowField::Rows(0)),
    (DevOp::GemmWideFp8, RowField::Rows(0)),
    (DevOp::GemmGluFp8, RowField::Rows(0)),
    (DevOp::GemmFp8Blk, RowField::Rows(0)),
    (DevOp::FlashMlaPrefill, RowField::Rows(4)),
    (DevOp::FlashMlaPrefillFp8, RowField::Rows(4)),
    (DevOp::MlaMergeFold, RowField::Rows(0)),
    (DevOp::MoeRouterTopkPf, RowField::Rows(4)),
    (DevOp::MoeAlignPf, RowField::Rows(0)),
    (DevOp::MoeCombinePf, RowField::Rows(2)),
    (DevOp::XReduce, RowField::RowsTimes(0)),
    (DevOp::XReduceTwoShot, RowField::RowsTimes(0)),
];

pub(crate) fn prefill_row_field(op: u16) -> Option<RowField> {
    PREFILL_ROW_FIELDS
        .iter()
        .find(|(o, _)| *o as u16 == op)
        .map(|&(_, f)| f)
}

/// [`rebase_chunk`], plus the RAGGED-M row shrink when `bucket` is `Some(T)`.
///
/// # What the shrink is for
///
/// A prefill bucket program is compiled at a fixed row count `T`, and the last
/// chunk of a prompt almost never has `T` real tokens. Without the shrink the
/// kernels compute all `T` rows and the padded ones are simply never read, so a
/// 1-token remainder costs a full `T`-row pass — and, worse, `plan_chunks` must
/// then pick the SMALLEST bucket covering the remainder to keep that waste down,
/// which is what turns a 4097-token prompt into `[4096, 128]`: two launches, the
/// second paying the whole T-invariant cost of a 78-layer pass. Measured on the
/// shipped config: 4096 -> 720.2 ms, 4097 -> 951.2 ms, for ONE more token.
///
/// With the shrink the row count is a runtime operand, so one 8192-bucket launch
/// carries 4097 real rows at the cost of ~4097 rows and the second launch
/// disappears. Every kernel this touches already tiles over `M`/`n_tok` and
/// bounds its own loads by it (`d_gemm_t`: `tm = ceil(M/BM)`, `r < M` on every A
/// fetch; `d_flash_mla_prefill`: `n_work = n_batch*n_tok*n_grp*nsplit`), so this
/// is not a new kernel contract — it is the operand finally being told the truth.
/// Workgroups whose tile index falls past the shortened range still run the
/// interpreter and still signal their successor counters, so the counter DAG is
/// untouched.
///
/// # The one operand that must move WITH it
///
/// `in.kvlen`. `d_flash_mla_decode` derives `qpos = kv_len - n_tok + t`, so
/// shrinking `n_tok` without shrinking `kv_len` shifts every query in the chunk
/// down by the padding. [`AmdEngine::prefill_prepare`] uploads `c0 + clen`
/// instead of `c0 + ch` under the same flag, and both sites read
/// [`AmdEngine::ragged_bucket`] so the pair cannot drift.
///
/// # Why the "field must equal the bucket width" guard
///
/// The mapping opcode -> field is a static table, but a field that holds the row
/// count in one packet may hold something else in another (the lm_head `Gemv`'s
/// `M = 1`; a `PLOW_GLM_XR_BAND` row-band `Gemm`'s `M = T/kb`). Rewriting one of
/// those would be silent and wrong. So a `Rows` field is patched only when it is
/// EXACTLY `T`, and a `RowsTimes` field only when it is an exact multiple of `T`
/// — anything else is left alone, which is always SAFE because computing the
/// padded row count is exactly what the engine did before this axis existed: pad
/// rows produce values nothing reads (the lm_head samples row `clen - 1`, `n_kv`
/// bounds every KV read at `c0 + clen`, and no prefill op reduces across the
/// token axis).
///
/// [`AmdEngine::refuse_unraggable`] refuses the one configuration where "left
/// alone" is NOT enough — row-banded collectives, where the band `Gemm` would be
/// skipped by the guard while its `XReduce` partner was rescaled — so the guard
/// never half-applies in silence.
pub(crate) fn rebase_chunk_rows(
    insts: &mut [DevInst64],
    names: &[String],
    c0: u32,
    clen: u32,
    t: u32,
    bucket: Option<u32>,
) {
    for d in insts.iter_mut() {
        let op = d.op;
        if let Some(f) = kv_write_row_field(op, names.get(d.t[0] as usize)) {
            d.i[f] = c0;
        }
        if (op == DevOp::HeadNormRope as u16 || op == DevOp::HeadNormRopeFp8 as u16) && d.fj[1] != 0
        {
            d.i[3] = c0;
        } else if op == DevOp::FlashPrefill as u16 || op == DevOp::FlashPrefillFp8 as u16 {
            d.i[4] = c0;
            d.i[1] = c0 + clen;
        } else if KDA_ROW_COUNT_OPS.iter().any(|&k| op == k as u16) {
            d.i[0] = clen;
        } else if op == DevOp::DsaPoolCompress as u16 {
            let pool_size = d.i[1].max(1);
            d.i[0] = clen / pool_size;
        } else if op == DevOp::DsaPoolStash as u16 {
            let pool_size = d.i[0].max(1);
            let pad = t.saturating_sub(clen);
            let complete = (clen / pool_size) * pool_size;
            match d.i[2].checked_sub(pad) {
                Some(row) if row >= complete && row < clen => d.i[2] = row,
                _ => d.op = DevOp::Nop as u16,
            }
        }
        let Some(t) = bucket.filter(|&t| t > 0 && clen < t) else {
            continue;
        };
        match prefill_row_field(op) {
            Some(RowField::Rows(f)) if d.i[f] == t => d.i[f] = clen,
            Some(RowField::RowsTimes(f)) if d.i[f] > 0 && d.i[f] % t == 0 => {
                d.i[f] = (d.i[f] / t) * clen
            }
            _ => {}
        }
    }
}
