//! The emit-time `ctx` as a CEILING, and the rule that narrows a packet to a
//! smaller LIVE bound at load.
//!
//! A packet's `ctx` reaches the device in exactly two shapes:
//!
//! 1. **Declared bytes** of the tensors whose extent is the context — the MLA
//!    latent caches, the indexer key cache and its scratch, the position/id
//!    staging arrays and the RoPE tables. Each is EXACTLY LINEAR in `ctx`
//!    ([`Scaling::Linear`]) or is a RoPE recipe whose row count IS `ctx`
//!    ([`Scaling::Rope`]).
//! 2. **Instruction fields** carrying the per-slot cache row stride, which is
//!    the same number: a `[slot][ctx][d]` cache addresses row `p` of slot `s` at
//!    `s*ctx + p`.
//!
//! Nothing else in the GLM/MLA emit moves with `ctx` over the range the runtime
//! will narrow across — `devgen::mla::ctx_bound_tests` pins that by emitting the
//! whole production recipe at two contexts and asserting the only differences
//! are the two shapes above, then by narrowing a 1M-ceiling emit and asserting
//! it equals a native emit at the live bound, instruction for instruction.
//!
//! The rewrite is directed by (OPCODE, FIELD), not by value. Scanning for the
//! ceiling's value is what one would write first and it is wrong: a 1048576
//! ceiling collides with an `XAllGather` element count, and rewriting that is a
//! silent corruption with no failure mode. [`STRIDE_SITES`] is the list, and
//! [`residual_ceiling`] is the backstop for the other direction — an op that
//! addresses a narrowed cache and keeps the ceiling in a slot this module does
//! not rewrite would stride into a buffer that is no longer that long, so the
//! packet is refused instead.

use crate::dev::{DevInst, DevInst64, DevOp, TENSOR_NONE, TENSOR_NONE16};
use crate::rope::{GenTensor, GEN_ROPE_COS, GEN_ROPE_IDX_COS, GEN_ROPE_IDX_SIN, GEN_ROPE_SIN};

/// An integer operand slot, in both instruction forms. `J` is the spare pair the
/// attention family carries the cache stride in; on the wire it shares a word
/// with `f[1]` (`DevInst64::fj[1]`), which no op reading `j[0]` uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Slot {
    I(usize),
    J(usize),
}

impl Slot {
    fn get(self, d: &DevInst) -> u32 {
        match self {
            Slot::I(k) => d.i[k],
            Slot::J(k) => d.j[k],
        }
    }
    fn set(self, d: &mut DevInst, v: u32) {
        match self {
            Slot::I(k) => d.i[k] = v,
            Slot::J(k) => d.j[k] = v,
        }
    }
    fn get64(self, d: &DevInst64) -> u32 {
        match self {
            Slot::I(k) => d.i[k],
            Slot::J(k) => d.fj[k + 1],
        }
    }
    fn set64(self, d: &mut DevInst64, v: u32) {
        match self {
            Slot::I(k) => d.i[k] = v,
            Slot::J(k) => d.fj[k + 1] = v,
        }
    }
}

/// Every site the MLA/DSA emitter writes the per-slot cache row stride into.
///
/// Read `crates/devgen/src/mla.rs` for the other end of this list; an op appears
/// once per slot it can carry the stride in (the rope/latent writers carry it in
/// `j[0]` on the decode-shaped path and in `i[7]` on the packed-prefill one, and
/// which of the two is live is a property of the emitted program, not of the
/// opcode — so both are listed and only the slot actually holding the ceiling is
/// rewritten).
pub const STRIDE_SITES: &[(DevOp, Slot)] = &[
    (DevOp::RmsNorm, Slot::I(7)),
    (DevOp::HeadNormRope, Slot::I(7)),
    (DevOp::HeadNormRope, Slot::J(0)),
    (DevOp::HeadNormRopeFp8, Slot::I(7)),
    (DevOp::HeadNormRopeFp8, Slot::J(0)),
    (DevOp::FlashMlaDecode, Slot::I(2)),
    (DevOp::FlashMlaPrefill, Slot::I(2)),
    (DevOp::FlashGatherDecode, Slot::I(2)),
    (DevOp::FlashMlaDecodeFp8, Slot::I(2)),
    (DevOp::FlashMlaPrefillFp8, Slot::I(2)),
    (DevOp::FlashDecode, Slot::I(3)),
    (DevOp::FlashDecodeFp8, Slot::I(3)),
    (DevOp::FlashPrefill, Slot::J(0)),
    (DevOp::FlashPrefillFp8, Slot::J(0)),
    (DevOp::IndexScore, Slot::I(2)),
    (DevOp::IndexSelect, Slot::I(0)),
    (DevOp::IndexScorePf, Slot::I(2)),
    (DevOp::IndexSelectPf, Slot::I(2)),
    (DevOp::IndexUnionPf, Slot::I(2)),
    (DevOp::IndexTpPf, Slot::I(1)),
];

fn sites_for(op: u16) -> impl Iterator<Item = Slot> {
    STRIDE_SITES
        .iter()
        .filter(move |(o, _)| *o as u16 == op)
        .map(|(_, s)| *s)
}

/// How one declared tensor's size depends on the emit-time `ctx`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scaling {
    /// `bytes = k * ctx` for an integer `k`. Narrowing multiplies by `to/from`.
    Linear,
    /// A RoPE table materialised from a [`GenTensor`] whose `ctx` field is the
    /// row count. Narrowing rewrites the recipe, and the bytes follow from it.
    Rope,
    /// Independent of `ctx` — a weight, a per-step scalar, an activation sized
    /// by the program's rows, or a cache sized by a window rather than a
    /// context.
    Inert,
    /// A per-sequence cache this module has no rule for: a `kv.` tensor whose
    /// layout is not `[slot][ctx][d]`, so neither leaving it nor scaling it is
    /// known to be right. Makes the packet unnarrowable.
    Unknown,
}

/// `kv.{l}.` suffixes whose cache is `[slot][ctx][d]`.
///
/// `kidx_pool`/`kidx_pool_scale` are deliberately ABSENT and therefore
/// [`Scaling::Unknown`]: the pooled indexer (`index_kpool > 1`) sizes them
/// `ceil(ctx / index_kpool)` and puts that pool COUNT, not `ctx`, in
/// `IndexScoreKpool`'s `i[2]` — so the count is not a stride site, would not be
/// rewritten, and [`residual_ceiling`] would not see it either. Refusing the
/// whole packet is the only honest answer until the pooled chain is in the test
/// above; nothing shipping sets `index_kpool`.
const KV_LINEAR: [&str; 4] = ["ckv", "krot", "scale", "kidx"];

/// `kv.{l}.` suffixes with a FIXED extent — the pooled indexer's decode ring is
/// `[index_kpool][di]`, a knob rather than a context.
const KV_INERT: [&str; 2] = ["kidx_ring", "kidx_ring_score"];

/// `act.` scratch sized by the context rather than by the program's rows.
const ACT_LINEAR: [&str; 5] = ["iscore", "iscore_pf", "iumask", "ibits", "icand"];

const IN_LINEAR: [&str; 2] = ["in.ids", "in.pos"];
const IN_ROPE: [&str; 8] = [
    "in.cos",
    "in.sin",
    "in.icos",
    "in.isin",
    "in.cos_full",
    "in.sin_full",
    "in.cos_slide",
    "in.sin_slide",
];

/// How `name`'s declared bytes move with the emit-time `ctx`.
///
/// `kv.` is the only prefix that can answer [`Scaling::Unknown`], and that is
/// deliberate: an unrecognised activation or input is at worst left at the
/// ceiling — memory held, never a wrong address — while an unrecognised
/// per-sequence cache means the packet is not the `[slot][ctx][d]` family this
/// module describes at all.
pub fn tensor_scaling(name: &str) -> Scaling {
    if IN_LINEAR.contains(&name) {
        return Scaling::Linear;
    }
    if IN_ROPE.contains(&name) {
        return Scaling::Rope;
    }
    if let Some(rest) = name.strip_prefix("act.") {
        // `act.x@band2048` and friends are one rank's row band of an activation:
        // sized by the program's rows, never by the context.
        let base = rest.split('@').next().unwrap_or(rest);
        return match ACT_LINEAR.contains(&base) {
            true => Scaling::Linear,
            false => Scaling::Inert,
        };
    }
    let Some(rest) = name.strip_prefix("kv.") else {
        return Scaling::Inert;
    };
    // K3's snapshot ring, sized by the widest PREFILL bucket.
    if rest == "blkres" {
        return Scaling::Inert;
    }
    match rest.split_once('.') {
        Some((layer, suffix)) if layer.bytes().all(|b| b.is_ascii_digit()) => {
            if KV_LINEAR.contains(&suffix) {
                Scaling::Linear
            } else if KV_INERT.contains(&suffix) {
                Scaling::Inert
            } else {
                Scaling::Unknown
            }
        }
        _ => Scaling::Unknown,
    }
}

/// Does `name` name a per-sequence cache this module narrows? The instructions
/// that address one must carry a stride site — see [`check_stride_coverage`].
pub fn is_narrowed_cache(name: &str) -> bool {
    name.starts_with("kv.") && tensor_scaling(name) == Scaling::Linear
}

/// `bytes` re-declared at the live bound `to`, for a [`Scaling::Linear`] tensor
/// the packet declared at `from`.
///
/// `None` when `bytes` is not an exact multiple of `from`: the tensor is not
/// `k * ctx` after all, so this module's rule does not describe it and the
/// caller must refuse the narrow rather than round.
pub fn linear_bytes(bytes: u64, from: u32, to: u32) -> Option<u64> {
    let (from, to) = (u64::from(from), u64::from(to));
    (from != 0 && bytes % from == 0).then(|| bytes / from * to)
}

/// Is `g` a RoPE recipe whose `ctx` field is the table's row count?
pub fn is_rope_recipe(g: &GenTensor) -> bool {
    matches!(
        g.kind,
        GEN_ROPE_COS | GEN_ROPE_SIN | GEN_ROPE_IDX_COS | GEN_ROPE_IDX_SIN
    )
}

/// Rewrite the stride sites of `insts` from `from` to `to` for instructions
/// whose referenced tensors match `matches_target`, returning how many moved.
pub fn restride_matching(
    insts: &mut [DevInst64],
    from: u32,
    to: u32,
    matches_target: impl Fn(u16) -> bool,
) -> usize {
    let mut n = 0;
    for d in insts.iter_mut() {
        if !d.t.iter().any(|&h| h != TENSOR_NONE16 && matches_target(h)) {
            continue;
        }
        for slot in sites_for(d.op) {
            if slot.get64(d) == from {
                slot.set64(d, to);
                n += 1;
                if matches!(
                    DevOp::from_u16(d.op),
                    Some(DevOp::FlashDecode | DevOp::FlashDecodeFp8)
                ) && d.fj[1] != 0
                    && from > 0
                    && d.fj[1] % from == 0
                {
                    d.fj[1] = (d.fj[1] / from) * to;
                }
            }
        }
    }
    n
}

/// Rewrite the stride sites of `insts` from `from` to `to`, returning how many
/// moved. A listed slot holding anything but `from` is left alone: the same
/// opcodes carry a WINDOW there on a sliding-attention packet, and a ring is not
/// a context.
pub fn restride(insts: &mut [DevInst64], from: u32, to: u32) -> usize {
    restride_matching(insts, from, to, |_| true)
}

/// [`restride`] on the builder-side instruction, where `j` is not yet packed
/// into `fj`. The emitter-side differential test works in this form.
pub fn restride_builder(insts: &mut [DevInst], from: u32, to: u32) -> usize {
    let mut n = 0;
    for d in insts.iter_mut() {
        for slot in sites_for(d.op) {
            if slot.get(d) == from {
                slot.set(d, to);
                n += 1;
                if matches!(
                    DevOp::from_u16(d.op),
                    Some(DevOp::FlashDecode | DevOp::FlashDecodeFp8)
                ) && d.j[0] != 0
                    && from > 0
                    && d.j[0] % from == 0
                {
                    d.j[0] = (d.j[0] / from) * to;
                }
            }
        }
    }
    n
}

/// The first instruction that addresses a narrowed cache and would keep the
/// CEILING in an operand [`restride`] does not rewrite, as `(index, opcode,
/// slot)`.
///
/// This is the soundness backstop for [`STRIDE_SITES`] being a hand-written
/// list. Shrinking `kv.{l}.ckv` from `[slot][from][d]` to `[slot][to][d]` while
/// some op still strides by `from` walks slot 1 off the end of the buffer, and
/// nothing on the device would report it — so a packet holding a stale ceiling
/// on a cache-addressing op is refused at load instead.
///
/// An op that addresses a cache and carries the ceiling NOWHERE is fine and is
/// not flagged: the prefill writers address a contiguous row run whose slot base
/// the host rebases from the tensor's own declared bytes, so they carry no
/// stride at all and shrink with it. `narrowed` answers whether a tensor handle
/// is one of the caches being shrunk.
pub fn residual_ceiling(
    insts: &[DevInst64],
    from: u32,
    narrowed: impl Fn(u16) -> bool,
) -> Option<(usize, u16, Slot)> {
    insts.iter().enumerate().find_map(|(k, d)| {
        if !d.t.iter().any(|&h| h != TENSOR_NONE16 && narrowed(h)) {
            return None;
        }
        let slots = (0..8).map(Slot::I).chain((0..2).map(Slot::J));
        let stale = slots.filter(|s| s.get64(d) == from);
        stale
            .filter(|s| !sites_for(d.op).any(|x| x == *s))
            .map(|s| (k, d.op, s))
            .next()
    })
}

/// [`residual_ceiling`] in the builder-side form.
pub fn residual_ceiling_builder(
    insts: &[DevInst],
    from: u32,
    narrowed: impl Fn(u32) -> bool,
) -> Option<(usize, u16, Slot)> {
    insts.iter().enumerate().find_map(|(k, d)| {
        if !d.t.iter().any(|&h| h != TENSOR_NONE && narrowed(h)) {
            return None;
        }
        let slots = (0..8).map(Slot::I).chain((0..2).map(Slot::J));
        let stale = slots.filter(|s| s.get(d) == from);
        stale
            .filter(|s| !sites_for(d.op).any(|x| x == *s))
            .map(|s| (k, d.op, s))
            .next()
    })
}

/// Backstop for cache rewrites: any op addressing a rewritten cache that still
/// holds `from - 1` (a stale mask companion) in ANY operand slot is flagged.
///
/// On full layers, masks are emitted as `u32::MAX`. On context-clamped rings or
/// un-widened masks, slots like `FlashDecode I(7)`, `FlashPrefill J(1)`, or
/// `HeadNormRope J(1)` hold `from - 1`. If any op addressing a rewritten cache
/// still holds `from - 1`, it was sized for the old bound and widening would
/// wrap or mask incorrectly.
pub fn residual_mask(
    insts: &[DevInst64],
    from: u32,
    rewritten: impl Fn(u16) -> bool,
) -> Option<(usize, u16, Slot)> {
    if from == 0 {
        return None;
    }
    let stale = from - 1;
    insts.iter().enumerate().find_map(|(k, d)| {
        if !d.t.iter().any(|&h| h != TENSOR_NONE16 && rewritten(h)) {
            return None;
        }
        let slots = (0..8).map(Slot::I).chain((0..2).map(Slot::J));
        slots
            .filter(|s| s.get64(d) == stale)
            .map(|s| (k, d.op, s))
            .next()
    })
}

/// [`residual_mask`] in the builder-side form.
pub fn residual_mask_builder(
    insts: &[DevInst],
    from: u32,
    rewritten: impl Fn(u32) -> bool,
) -> Option<(usize, u16, Slot)> {
    if from == 0 {
        return None;
    }
    let stale = from - 1;
    insts.iter().enumerate().find_map(|(k, d)| {
        if !d.t.iter().any(|&h| h != TENSOR_NONE && rewritten(h)) {
            return None;
        }
        let slots = (0..8).map(Slot::I).chain((0..2).map(Slot::J));
        slots
            .filter(|s| s.get(d) == stale)
            .map(|s| (k, d.op, s))
            .next()
    })
}

