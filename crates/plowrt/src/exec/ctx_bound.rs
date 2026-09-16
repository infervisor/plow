//! Narrow a loaded packet from the context CEILING it was emitted at to the
//! LIVE bound this server will serve.
//!
//! `plowc` bakes a `ctx` into the packet, and until now that number was the
//! serving length: the KV caches are `[slot][ctx][d]`, so a packet emitted at
//! 81920 with a ladder topping at 32 reserves `81920 * 32 * 55296 B` = 135 GiB
//! of latent cache per rank whether or not anyone asks for an 80k prompt, and
//! that plus 93.4 GiB of weights does not fit a 192 GiB MI300X. The only lever
//! was a NARROWER LADDER — trading concurrency for context, at emit time, in a
//! packet that then serves exactly one context and cannot be compared against
//! the packet tuned for another.
//!
//! So the emitted `ctx` becomes a CEILING and the serving bound moves here.
//! `PLOW_LIVE_CTX` re-declares every ctx-scaled tensor and rewrites every cache
//! stride at load ([`packet::ctx_bound`]), which is the same packet `plowc`
//! would have written at that bound — `devgen::mla::ctx_bound_tests` narrows a
//! 1M-ceiling GLM emit to 81920 and asserts it equals a native 81920 emit,
//! instruction for instruction. Everything downstream reads `max_ctx` out of
//! `in.pos` and therefore needs no change at all: the narrow happens before the
//! blob is looked at.
//!
//! WHAT IS NOT NARROWED, and why this is not a full `emit(live)`: the emitter's
//! POLICY terms stay at their ceiling values. Over `live >= 16384` that is a
//! distinction without a difference — `glm_nsplit` is pinned at its measured
//! ceiling of 64 from 16384 up, `glm_gf` from 4096 up, and `glm_dsa_pf_cap` /
//! `glm_dsa_select_width` from 16384 / 2048 up — so the narrowed packet IS the
//! native one. Below 16384 two of them stop saturating and the packet keeps a
//! split count and a union capacity sized for the ceiling: spare capacity in the
//! second case, and in the first a measured decode cost (`ns=64` at live 8192 is
//! 81.3 us/layer against 73.3) that `PLOW_MLA_NS_LIVE=1` recovers per step.
//! `GlmCfg::dsa`'s 65536 crossover likewise stays armed below it, so a narrowed
//! packet attends sparsely where a native emit at that bound would have gone
//! dense. None of the three can produce a wrong address, which is why they are a
//! warning and the geometry checks below are refusals.

use crate::asset::devblob::DevBlob;
use crate::{Result, RuntimeError};
use packet::ctx_bound::{self, Scaling};

/// Below this the emitter's `min(_, ctx)` policy terms stop saturating fast
/// enough to mean anything: `glm_dsa_select_width` is `min(index_topk, ctx)` and
/// `index_topk` is 2048, so a packet narrowed under it carries a selection width
/// wider than the cache it selects from.
const MIN_LIVE_CTX: u32 = 2048;

/// The bound under which the narrowed packet stops being the packet `plowc`
/// would have emitted. See the module header.
const POLICY_SATURATION_CTX: u32 = 16384;

/// The context a packet was emitted at, from the only tensor whose extent is
/// unambiguously it. This is the same derivation `AmdEngine::max_ctx` uses, and
/// it must stay that way: the whole point is that narrowing `in.pos` is what
/// moves the engine's bound.
fn declared_ctx(blob: &DevBlob) -> Option<u32> {
    let t = blob.tensors.iter().find(|t| t.name == "in.pos")?;
    u32::try_from(t.bytes / 4).ok()
}

/// Re-declare `blob` at the live bound `want`, returning the bound in force.
///
/// A no-op (and `Ok(ceiling)`) when `want` is at or above what the packet was
/// emitted at: the ceiling is a ceiling, not a target.
pub(crate) fn narrow(blob: &mut DevBlob, want: u32) -> Result<u32> {
    let err = |s: String| RuntimeError::Device(s);
    let ceiling = declared_ctx(blob).ok_or_else(|| {
        err("PLOW_LIVE_CTX: this packet declares no `in.pos`, so it carries no context to \
             narrow".into())
    })?;
    if want >= ceiling {
        return Ok(ceiling);
    }
    if want < MIN_LIVE_CTX {
        return Err(err(format!(
            "PLOW_LIVE_CTX={want} is below the {MIN_LIVE_CTX}-token floor: the emitter's \
             selection width is min(index_topk, ctx) and index_topk is 2048, so a narrower \
             bound leaves the packet selecting more rows than its cache holds"
        )));
    }
    let widest = blob.progs.iter().map(|p| p.t).max().unwrap_or(1);
    if want < widest {
        return Err(err(format!(
            "PLOW_LIVE_CTX={want} is narrower than this packet's widest prefill bucket \
             ({widest} rows). `in.ids`/`in.pos` are staged a whole chunk at a time, so the \
             bound cannot be below the largest chunk the packet can run"
        )));
    }

    // Which tensors move, and can they. A `kv.` cache with no rule is the hard
    // stop: it is a per-sequence buffer whose layout is not `[slot][ctx][d]`, so
    // neither leaving it nor scaling it is known to be right.
    let mut plan: Vec<(usize, u64)> = Vec::new();
    for (h, t) in blob.tensors.iter().enumerate() {
        let scaling = ctx_bound::tensor_scaling(&t.name);
        if scaling == Scaling::Inert {
            continue;
        }
        if scaling == Scaling::Unknown {
            return Err(err(format!(
                "PLOW_LIVE_CTX: `{}` is a per-sequence cache `packet::ctx_bound` has no \
                 rule for (the pooled indexer's caches are one; so is every sliding-window \
                 ring). Narrowing it would be a guess at its layout — serve this packet at \
                 its emitted ceiling",
                t.name
            )));
        }
        if t.init.is_some() {
            return Err(err(format!(
                "PLOW_LIVE_CTX: `{}` is ctx-scaled but carries BAKED init bytes, which are \
                 sized for the ceiling. Re-emit with the RoPE tables as recipes (the \
                 default; `--no-rope-gen` bakes them)",
                t.name
            )));
        }
        if scaling == Scaling::Rope {
            // A table whose recipe is missing is one this pass cannot re-cut,
            // and leaving it at the ceiling would leave the engine's bind-time
            // length check disagreeing with the declaration.
            if !blob
                .gen
                .iter()
                .any(|g| g.tensor as usize == h && ctx_bound::is_rope_recipe(g))
            {
                return Err(err(format!(
                    "PLOW_LIVE_CTX: `{}` is a ctx-sized RoPE table with no recipe to re-cut \
                     it from",
                    t.name
                )));
            }
            continue;
        }
        let bytes = ctx_bound::linear_bytes(t.bytes, ceiling, want).ok_or_else(|| {
            err(format!(
                "PLOW_LIVE_CTX: `{}` is {} B, not a multiple of the emitted ctx {ceiling} — \
                 it is not `k * ctx` and this module's rule does not describe it",
                t.name, t.bytes
            ))
        })?;
        plan.push((h, bytes));
    }
    for g in &blob.gen {
        let name = &blob.tensors[g.tensor as usize].name;
        if ctx_bound::is_rope_recipe(g) && ctx_bound::tensor_scaling(name) != Scaling::Rope {
            return Err(err(format!(
                "PLOW_LIVE_CTX: `{name}` carries a RoPE recipe but is not one of the tables \
                 `packet::ctx_bound` narrows, so its rows would stay sized for the ceiling"
            )));
        }
    }

    // Nothing may keep the ceiling in an operand the rewrite does not reach.
    let narrowed: std::collections::HashSet<u16> = blob
        .tensors
        .iter()
        .enumerate()
        .filter(|(_, t)| ctx_bound::is_narrowed_cache(&t.name))
        .filter_map(|(h, _)| u16::try_from(h).ok())
        .collect();
    for p in &blob.progs {
        if let Some((k, op, slot)) =
            ctx_bound::residual_ceiling(&p.insts, ceiling, |h| narrowed.contains(&h))
        {
            return Err(err(format!(
                "PLOW_LIVE_CTX: program T={} instruction {k} (op {op}) addresses a narrowed \
                 cache and holds the emitted ctx {ceiling} in {slot:?}, which is not a \
                 stride site `packet::ctx_bound` rewrites. Narrowing would leave it striding \
                 past the end of the cache",
                p.t
            )));
        }
    }

    // Commit: recipes first, so a RoPE table's bytes follow from the recipe the
    // engine will actually run rather than from a second copy of its formula.
    for g in &mut blob.gen {
        if ctx_bound::is_rope_recipe(g) {
            g.ctx = want;
        }
    }
    for g in &blob.gen {
        if ctx_bound::is_rope_recipe(g) {
            blob.tensors[g.tensor as usize].bytes = g.byte_len();
        }
    }
    for (h, bytes) in plan {
        blob.tensors[h].bytes = bytes;
    }
    let mut sites = 0;
    for p in &mut blob.progs {
        sites += ctx_bound::restride(&mut p.insts, ceiling, want);
    }
    debug_assert_eq!(declared_ctx(blob), Some(want));

    if want < POLICY_SATURATION_CTX {
        tracing::warn!(
            live = want, ceiling,
            "PLOW_LIVE_CTX below {POLICY_SATURATION_CTX}: the packet keeps the MLA split \
             count, the DSA union capacity and the sparse-decode arm the emitter chose for \
             its ceiling. Set PLOW_MLA_NS_LIVE=1 to track the split count from the live \
             kv_len"
        );
    }
    tracing::info!(
        live = want, ceiling, stride_sites = sites,
        "PLOW_LIVE_CTX: packet re-declared at the live context bound"
    );
    Ok(want)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asset::devblob::{DevProg, DevTensor};
    use packet::dev::{DevInst64, DevOp, TENSOR_NONE16};
    use packet::rope::{GenTensor, RopeScale};

    const CEILING: u32 = 8192;
    const BATCH: u64 = 2;

    /// An MLA-shaped packet: the two staging arrays, a RoPE recipe, a latent
    /// cache and its rope cache, a weight, and a decode program that writes the
    /// latent (stride in `j[0]`) and reads it (stride in `i[2]`).
    ///
    /// `XAllGather` carries the ceiling as an ELEMENT COUNT and touches no
    /// cache: it is the collision a value-directed rewrite would corrupt, and it
    /// is in the fixture so that staying untouched is a test rather than a hope.
    fn mla_blob(ctx: u32) -> DevBlob {
        let [cos, _] = GenTensor::rope_pair(ctx, 64, 10000.0, 1.0, RopeScale::None);
        let t = |name: &str, bytes: u64| DevTensor {
            name: name.into(),
            bytes,
            init: None,
        };
        let tensors = vec![
            t("in.ids", ctx as u64 * 4),
            t("in.pos", ctx as u64 * 4),
            t("in.cos", cos.byte_len()),
            t("kv.0.ckv", BATCH * ctx as u64 * 512),
            t("kv.0.krot", BATCH * ctx as u64 * 128),
            t("layers.0.wq", 4096),
        ];
        let none = TENSOR_NONE16;
        let inst = |op: DevOp, t0: u16, i: [u32; 8], fj: [u32; 3]| DevInst64 {
            op: op as u16,
            blocks: 1,
            fj,
            t: [t0, none, none, none, 3, 4, none, none],
            i,
        };
        let insts = vec![
            // Latent writer: stride in j[0], which is `fj[1]` on the wire.
            inst(
                DevOp::HeadNormRopeFp8,
                3,
                [1, 1, 512, 0, 0, 0, 1, 0],
                [0, ctx, 0],
            ),
            // Flash decode: stride in i[2].
            inst(
                DevOp::FlashMlaDecodeFp8,
                0,
                [1, 8, ctx, 0, 64, 0, 0, 4],
                [0, 0, 0],
            ),
            DevInst64 {
                op: DevOp::XAllGather as u16,
                blocks: 1,
                fj: [0; 3],
                t: [0; 8],
                i: [4096, ctx, 0, 0, 0, 0, 0, 0],
            },
        ];
        DevBlob {
            n_cu: 1,
            flags: 0,
            target: 0,
            tensors,
            init: Vec::new(),
            kvrow: Vec::new(),
            progs: vec![DevProg {
                t: 1,
                role: packet::devbuild::ProgramRole::DecodeRung { rows: 1 },
                n_counter: 0,
                insts,
                stream: Vec::new(),
                stream_ofs: Vec::new(),
                stream_len: Vec::new(),
                waits: Vec::new(),
                succs: Vec::new(),
                gq_stream: Vec::new(),
                gq_seg_ofs: Vec::new(),
                l2_domains: 0,
            }],
            sections: Vec::new(),
            gen: vec![GenTensor { tensor: 2, ..cos }],
            tp: None,
            parent: None,
        }
    }

    fn bytes_of(b: &DevBlob, name: &str) -> u64 {
        b.tensors.iter().find(|t| t.name == name).unwrap().bytes
    }

    #[test]
    fn narrowing_rescales_the_caches_the_recipes_and_the_strides() {
        let mut b = mla_blob(CEILING);
        assert_eq!(narrow(&mut b, 2048).unwrap(), 2048);
        assert_eq!(
            bytes_of(&b, "in.pos"),
            2048 * 4,
            "the engine reads max_ctx out of this one"
        );
        assert_eq!(bytes_of(&b, "in.ids"), 2048 * 4);
        assert_eq!(bytes_of(&b, "kv.0.ckv"), BATCH * 2048 * 512);
        assert_eq!(bytes_of(&b, "kv.0.krot"), BATCH * 2048 * 128);
        assert_eq!(bytes_of(&b, "layers.0.wq"), 4096, "a weight is not a context");
        // The recipe moved with the declaration, so the bind-time expansion
        // still produces exactly the declared bytes.
        assert_eq!(b.gen[0].ctx, 2048);
        assert_eq!(bytes_of(&b, "in.cos"), b.gen[0].byte_len());
        let insts = &b.progs[0].insts;
        assert_eq!(insts[0].fj[1], 2048, "latent writer out_stride");
        assert_eq!(insts[1].i[2], 2048, "flash kv_stride");
        assert_eq!(insts[2].i[1], CEILING, "an all-gather count is not a stride");
    }

    /// `d = identity`: a bound at or above the emitted ceiling changes nothing,
    /// which is the path every packet in the tree takes today.
    #[test]
    fn a_bound_at_or_above_the_ceiling_is_the_identity() {
        let before = mla_blob(CEILING);
        for want in [CEILING, CEILING * 4] {
            let mut b = mla_blob(CEILING);
            assert_eq!(narrow(&mut b, want).unwrap(), CEILING);
            assert_eq!(b.tensors.len(), before.tensors.len());
            for (x, y) in b.tensors.iter().zip(&before.tensors) {
                assert_eq!((&x.name, x.bytes), (&y.name, y.bytes));
            }
            assert_eq!(b.gen, before.gen);
            assert_eq!(b.progs[0].insts, before.progs[0].insts);
        }
    }

    /// A cache-addressing op holding the ceiling somewhere the rewrite does not
    /// reach would stride past the end of the shrunk cache, with no fault and no
    /// wrong-looking output. Refuse, and commit nothing.
    #[test]
    fn a_stale_ceiling_on_a_cache_op_is_refused() {
        let mut b = mla_blob(CEILING);
        b.progs[0].insts[1].i[3] = CEILING;
        let e = narrow(&mut b, 2048).unwrap_err().to_string();
        assert!(e.contains("I(3)"), "{e}");
        assert_eq!(
            bytes_of(&b, "kv.0.ckv"),
            BATCH * CEILING as u64 * 512,
            "a refused narrow leaves the packet alone"
        );
    }

    #[test]
    fn a_kv_cache_with_no_scaling_rule_is_refused() {
        let mut b = mla_blob(CEILING);
        b.tensors[3].name = "kv.0.mystery".into();
        let e = narrow(&mut b, 2048).unwrap_err().to_string();
        assert!(e.contains("kv.0.mystery"), "{e}");
    }

    /// The staging arrays hold a whole prefill chunk, so the bound cannot go
    /// under the widest bucket the packet can run.
    #[test]
    fn a_bound_under_the_widest_prefill_bucket_is_refused() {
        let mut b = mla_blob(CEILING);
        b.progs[0].t = 4096;
        let e = narrow(&mut b, 2048).unwrap_err().to_string();
        assert!(e.contains("4096"), "{e}");
    }

    #[test]
    fn a_bound_under_the_selection_width_is_refused() {
        let mut b = mla_blob(CEILING);
        let e = narrow(&mut b, 1024).unwrap_err().to_string();
        assert!(e.contains("floor"), "{e}");
    }

    /// Baked RoPE bytes are sized for the ceiling and cannot be re-cut here.
    #[test]
    fn a_baked_rope_table_is_refused() {
        let mut b = mla_blob(CEILING);
        b.gen.clear();
        b.tensors[2].init = Some(0..1);
        let e = narrow(&mut b, 2048).unwrap_err().to_string();
        assert!(e.contains("in.cos"), "{e}");
    }

    /// A RoPE table under a name this pass does not narrow would keep the
    /// ceiling's rows while the strides around it shrank.
    #[test]
    fn a_rope_recipe_on_an_unnarrowed_tensor_is_refused() {
        let mut b = mla_blob(CEILING);
        b.tensors[2].name = "rope.cos".into();
        let e = narrow(&mut b, 2048).unwrap_err().to_string();
        assert!(e.contains("rope.cos"), "{e}");
    }
}
