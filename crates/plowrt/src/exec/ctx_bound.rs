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
use packet::dev::DevOp;

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
/// it must stay that way: the whole point is that narrowing/widening `in.pos` is what
/// moves the engine's bound.
pub fn declared_ctx(blob: &DevBlob) -> Option<u32> {
    let t = blob.tensors.iter().find(|t| t.name == "in.pos")?;
    u32::try_from(t.bytes / 4).ok()
}

/// Re-declare `blob` at the live bound `want`, returning the bound in force.
///
/// If `want == ceiling`, returns `Ok(ceiling)`.
/// If `want < ceiling`, narrows the packet.
/// If `want > ceiling`, widens the packet.
pub fn rescale(
    blob: &mut DevBlob,
    want: u32,
    manifest: Option<&plow_asset::live_kv::Manifest>,
) -> Result<u32> {
    let err = |s: String| RuntimeError::Device(s);
    let ceiling = declared_ctx(blob).ok_or_else(|| {
        err("PLOW_LIVE_CTX: this packet declares no `in.pos`, so it carries no context to \
             rescale".into())
    })?;
    if want == ceiling {
        return Ok(ceiling);
    }

    let derived_manifest = if manifest.is_none() {
        blob.with_packet_view(plow_asset::live_kv::emit).ok()
    } else {
        None
    };
    let active_manifest = manifest.or(derived_manifest.as_ref());

    if want < ceiling {
        // Narrowing checks
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
    } else {
        // Widening checks
        // 2.3 Refuse DSA/indexer: NV dense only in v1
        let has_indexer_tensors = blob
            .tensors
            .iter()
            .any(|t| t.name.contains("kidx") || t.name.contains("index"));
        let has_indexer_ops = blob.progs.iter().any(|p| {
            p.insts.iter().any(|d| {
                matches!(
                    DevOp::from_u16(d.op),
                    Some(
                        DevOp::IndexScore
                            | DevOp::IndexSelect
                            | DevOp::IndexScorePf
                            | DevOp::IndexSelectPf
                            | DevOp::IndexUnionPf
                            | DevOp::IndexTpPf
                    )
                )
            })
        });
        if has_indexer_tensors || has_indexer_ops {
            return Err(err(
                "PLOW_LIVE_CTX: widening DSA/indexer packets is unsupported in v1 (NV dense only)"
                    .into(),
            ));
        }

        // 2.2 Prefill chunk ladder
        let max_chunk = blob
            .progs
            .iter()
            .filter(|p| !p.role.is_decode_rung())
            .map(|p| p.t)
            .max()
            .unwrap_or(1);
        if max_chunk == ceiling {
            return Err(err(format!(
                "PLOW_LIVE_CTX={want}: this packet's widest prefill bucket ({max_chunk} rows) \
                 equals its emitted ctx, so the emitter's `ctx.min(max_chunk)` clamped the bucket \
                 ladder. Widening would keep prefilling in {max_chunk}-row chunks; re-emit at the \
                 wider ctx instead"
            )));
        }

        // 1c Ring/mask invariant
        if let Some(m) = active_manifest {
            for c in &m.caches {
                if c.window > 0 {
                    let ring_threshold = plow_asset::extension::kv_ring_rows(c.window, max_chunk);
                    if ceiling < ring_threshold {
                        return Err(err(format!(
                            "PLOW_LIVE_CTX={want}: widen is ring-unsafe because packet ctx_old {ceiling} < \
                             kv_ring_rows(window={}, chunk={max_chunk}) = {ring_threshold}: sliding ring would alias on wrap",
                            c.window
                        )));
                    }
                }
            }
        }

        // 1d RoPE warning
        tracing::warn!(
            live = want,
            ceiling,
            "PLOW_LIVE_CTX widened context past emitted ceiling"
        );
    }

    // Which tensors move, and can they. A `kv.` cache with no rule is the hard
    // stop: it is a per-sequence buffer whose layout is not `[slot][ctx][d]`, so
    // neither leaving it nor scaling it is known to be right.
    // 1b: Classify caches from the manifest, not the name.
    let mut plan: Vec<(usize, u64)> = Vec::new();
    let mut rewritten_caches: std::collections::HashSet<u16> = std::collections::HashSet::new();

    for (h, t) in blob.tensors.iter().enumerate() {
        let h16 = u16::try_from(h).ok();
        let cache_entry = active_manifest.and_then(|m| {
            m.caches.iter().find(|c| {
                h16.is_some_and(|h| c.pair.contains(&h) || c.scales.is_some_and(|s| s.contains(&h)))
            })
        });

        let scaling = if let Some(c) = cache_entry {
            if c.window == 0 {
                Scaling::Linear
            } else {
                Scaling::Inert
            }
        } else {
            ctx_bound::tensor_scaling(&t.name)
        };

        if scaling == Scaling::Inert {
            continue;
        }
        if scaling == Scaling::Unknown {
            return Err(err(format!(
                "PLOW_LIVE_CTX: `{}` is a per-sequence cache `packet::ctx_bound` has no \
                 rule for (the pooled indexer's caches are one; so is every sliding-window \
                 ring). Rescaling it would be a guess at its layout — serve this packet at \
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
        if let Some(h16) = h16 {
            if ctx_bound::is_narrowed_cache(&t.name)
                || cache_entry.is_some_and(|c| c.window == 0)
                || t.name.starts_with("kv.")
            {
                rewritten_caches.insert(h16);
            }
        }
    }

    for g in &blob.gen {
        let name = &blob.tensors[g.tensor as usize].name;
        if ctx_bound::is_rope_recipe(g) && ctx_bound::tensor_scaling(name) != Scaling::Rope {
            return Err(err(format!(
                "PLOW_LIVE_CTX: `{name}` carries a RoPE recipe but is not one of the tables \
                 `packet::ctx_bound` rescales, so its rows would stay sized for the ceiling"
            )));
        }
    }

    // Nothing may keep the ceiling in an operand the rewrite does not reach.
    for p in &blob.progs {
        if let Some((k, op, slot)) =
            ctx_bound::residual_ceiling(&p.insts, ceiling, |h| rewritten_caches.contains(&h))
        {
            return Err(err(format!(
                "PLOW_LIVE_CTX: program T={} instruction {k} (op {op}) addresses a rewritten \
                 cache and holds the emitted ctx {ceiling} in {slot:?}, which is not a \
                 stride site `packet::ctx_bound` rewrites. Rescaling would leave it striding \
                 past the end of the cache",
                p.t
            )));
        }
        // 1c Mask-companion backstop (widening only)
        if want > ceiling {
            if let Some((k, op, slot)) =
                ctx_bound::residual_mask(&p.insts, ceiling, |h| rewritten_caches.contains(&h))
            {
                return Err(err(format!(
                    "PLOW_LIVE_CTX: program T={} instruction {k} (op {op}) addresses a rewritten \
                     cache and holds stale mask {} (emitted ctx - 1) in {slot:?}. \
                     Widening would leave it masking or wrapping with the un-widened bound",
                    p.t, ceiling - 1
                )));
            }
        }
    }

    // Commit: recipes first, so a RoPE table's bytes follow from the recipe the
    // engine will actually run rather than from a second copy of its formula.
    for g in &mut blob.gen {
        if ctx_bound::is_rope_recipe(g) {
            g.ctx = want;
        } else if g.kind == packet::rope::GEN_TMAP_KV_PAIR {
            let pair = [g.aux as u16, g.scale as u16];
            let is_full = active_manifest.map_or(g.ctx == ceiling, |m| {
                m.caches.iter().any(|c| c.pair == pair && c.window == 0)
            });
            if is_full && g.ctx == ceiling {
                g.ctx = want;
            }
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
    let matches_cache = |h| rewritten_caches.is_empty() || rewritten_caches.contains(&h);
    let mut sites = 0;
    for p in &mut blob.progs {
        sites += ctx_bound::restride_matching(&mut p.insts, ceiling, want, matches_cache);
    }
    debug_assert_eq!(declared_ctx(blob), Some(want));

    if want < ceiling && want < POLICY_SATURATION_CTX {
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

/// Re-declare `blob` at the live bound `want` (narrowing), returning the bound in force.
pub fn narrow(blob: &mut DevBlob, want: u32) -> Result<u32> {
    let ceiling = declared_ctx(blob).ok_or_else(|| {
        RuntimeError::Device("PLOW_LIVE_CTX: this packet declares no `in.pos`".into())
    })?;
    if want >= ceiling {
        return Ok(ceiling);
    }
    rescale(blob, want, None)
}

/// Re-declare `blob` at the live bound `want` (widening), returning the bound in force.
pub fn widen(
    blob: &mut DevBlob,
    want: u32,
    manifest: Option<&plow_asset::live_kv::Manifest>,
) -> Result<u32> {
    let ceiling = declared_ctx(blob).ok_or_else(|| {
        RuntimeError::Device("PLOW_LIVE_CTX: this packet declares no `in.pos`".into())
    })?;
    if want <= ceiling {
        return Ok(ceiling);
    }
    rescale(blob, want, manifest)
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

    fn nv_dense_blob(ctx: u32) -> (DevBlob, plow_asset::live_kv::Manifest) {
        let [cos, _] = GenTensor::rope_pair(ctx, 64, 10000.0, 1.0, RopeScale::None);
        let t = |name: &str, bytes: u64| DevTensor {
            name: name.into(),
            bytes,
            init: None,
        };
        // 0: in.ids, 1: in.pos, 2: in.cos, 3: kv.0.k, 4: kv.0.v, 5: kv.1.k, 6: kv.1.v
        let tensors = vec![
            t("in.ids", ctx as u64 * 4),
            t("in.pos", ctx as u64 * 4),
            t("in.cos", cos.byte_len()),
            t("kv.0.k", ctx as u64 * 128),
            t("kv.0.v", ctx as u64 * 128),
            t("kv.1.k", 8192 * 128),
            t("kv.1.v", 8192 * 128),
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
            // Flash decode: kv_stride in i[3]
            inst(
                DevOp::FlashDecode,
                0,
                [1, 8, 0, ctx, 64, 0, 0, u32::MAX],
                [0, 0, 0],
            ),
            // Flash prefill: kv_stride in j[0] (= fj[1])
            inst(
                DevOp::FlashPrefill,
                0,
                [0; 8],
                [0, ctx, u32::MAX],
            ),
            // HeadNormRope: out_stride in j[0] (= fj[1])
            inst(
                DevOp::HeadNormRope,
                3,
                [0; 8],
                [0, ctx, u32::MAX],
            ),
        ];
        let blob = DevBlob {
            n_cu: 1,
            flags: 0,
            target: 0,
            tensors,
            init: Vec::new(),
            kvrow: Vec::new(),
            progs: vec![
                DevProg {
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
                },
                DevProg {
                    t: 4096,
                    role: packet::devbuild::ProgramRole::PrefillBucket { rows: 4096 },
                    n_counter: 0,
                    insts: Vec::new(),
                    stream: Vec::new(),
                    stream_ofs: Vec::new(),
                    stream_len: Vec::new(),
                    waits: Vec::new(),
                    succs: Vec::new(),
                    gq_stream: Vec::new(),
                    gq_seg_ofs: Vec::new(),
                    l2_domains: 0,
                },
            ],
            sections: Vec::new(),
            gen: vec![GenTensor { tensor: 2, ..cos }],
            tp: None,
            parent: None,
        };
        let manifest = plow_asset::live_kv::Manifest {
            version: 1,
            n_cu: 1,
            batch: 1,
            max_ctx: ctx,
            position: 1,
            kv_length: 0,
            caches: vec![
                plow_asset::live_kv::Cache {
                    pair: [3, 4],
                    scales: None,
                    heads: 1,
                    hd: 128,
                    stride: ctx,
                    window: 0,
                    mask: u32::MAX,
                },
                plow_asset::live_kv::Cache {
                    pair: [5, 6],
                    scales: None,
                    heads: 1,
                    hd: 128,
                    stride: 8192,
                    window: 1024,
                    mask: 8191,
                },
            ],
            maps: Vec::new(),
            programs: Vec::new(),
            splitk: Vec::new(),
        };
        (blob, manifest)
    }

    #[test]
    fn widening_rescales_nv_dense_blob_with_manifest() {
        let (mut b, m) = nv_dense_blob(16384);
        assert_eq!(widen(&mut b, 32768, Some(&m)).unwrap(), 32768);
        assert_eq!(bytes_of(&b, "in.pos"), 32768 * 4);
        assert_eq!(bytes_of(&b, "in.ids"), 32768 * 4);
        assert_eq!(bytes_of(&b, "kv.0.k"), 32768 * 128);
        assert_eq!(bytes_of(&b, "kv.0.v"), 32768 * 128);
        assert_eq!(bytes_of(&b, "kv.1.k"), 8192 * 128, "sliding ring is inert");
        assert_eq!(bytes_of(&b, "kv.1.v"), 8192 * 128, "sliding ring is inert");
        assert_eq!(b.gen[0].ctx, 32768);
        assert_eq!(bytes_of(&b, "in.cos"), b.gen[0].byte_len());

        let insts = &b.progs[0].insts;
        assert_eq!(insts[0].i[3], 32768, "FlashDecode kv_stride");
        assert_eq!(insts[1].fj[1], 32768, "FlashPrefill kv_stride");
        assert_eq!(insts[2].fj[1], 32768, "HeadNormRope out_stride");
    }

    #[test]
    fn widen_refuses_ring_unsafe_packet() {
        let (mut b, mut m) = nv_dense_blob(4096);
        b.progs[1].t = 2048;
        m.caches[1].window = 4096;
        let e = widen(&mut b, 16384, Some(&m)).unwrap_err().to_string();
        assert!(e.contains("ring-unsafe"), "{e}");
    }

    #[test]
    fn widen_refuses_residual_mask() {
        let (mut b, m) = nv_dense_blob(16384);
        b.progs[0].insts[0].i[7] = 16383;
        let e = widen(&mut b, 32768, Some(&m)).unwrap_err().to_string();
        assert!(e.contains("stale mask"), "{e}");
    }

    #[test]
    fn widen_refuses_dsa_indexer_packet() {
        let (mut b, m) = nv_dense_blob(16384);
        b.tensors[3].name = "kv.0.kidx".into();
        let e = widen(&mut b, 32768, Some(&m)).unwrap_err().to_string();
        assert!(e.contains("DSA/indexer"), "{e}");
    }

    #[test]
    fn widen_refuses_small_prefill_chunk_ladder() {
        let (mut b, m) = nv_dense_blob(4096);
        let e = widen(&mut b, 16384, Some(&m)).unwrap_err().to_string();
        assert!(e.contains("clamped the bucket ladder"), "{e}");
    }

    #[test]
    fn widen_accepts_unclamped_prefill_ladder() {
        let (mut b, m) = nv_dense_blob(16384);
        assert_eq!(widen(&mut b, 32768, Some(&m)).unwrap(), 32768);
    }
}
