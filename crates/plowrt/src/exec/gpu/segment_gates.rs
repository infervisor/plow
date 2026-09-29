//! Gates a segment launch boundary already satisfies.
//!
//! A segmented prefill program runs one launch per segment, in order, on one stream (or as
//! one captured chain). A wait on a producer in an EARLIER segment is therefore satisfied
//! before the consumer's launch starts, and a counter nobody in its own segment waits on is
//! never read. Dropping both removes, per work item, the counter poll and its acquire fence
//! and the release fence and counter bump: ~4 dependent global round trips of each light
//! segment's fixed cost.

use packet::dev::{StreamEnt, Wait, SE_FINE, SE_XCTR};

use crate::asset::devblob::DevProg;

/// Gate tables with only same-segment dependencies: `(stream, gq_stream, waits, succs)`.
pub(super) struct SegmentGates {
    pub(super) stream: Vec<StreamEnt>,
    pub(super) gq_stream: Vec<StreamEnt>,
    pub(super) waits: Vec<Wait>,
    pub(super) succs: Vec<u32>,
    pub(super) dropped_waits: usize,
    pub(super) dropped_succs: usize,
}

/// `None` when the program's gates are not the coarse one-counter-per-instruction form this
/// reasoning covers (then the packet's tables stay as they are). `waits` is the program's wait
/// table after any host rewrite (library producers' thresholds zeroed).
pub(super) fn segment_local(g: &DevProg, waits: &[Wait]) -> Option<SegmentGates> {
    let n = g.insts.len();
    if g.n_counter as usize != n || g.gq_stream.is_empty() {
        return None;
    }
    let mut placement: Vec<Option<u16>> = vec![None; n];
    for e in g.gq_stream.iter().chain(&g.stream) {
        let slot = placement.get_mut(e.inst as usize)?;
        if slot.is_some_and(|s| s != e.seg) || e.flags & (SE_FINE | SE_XCTR) != 0 {
            return None;
        }
        *slot = Some(e.seg);
        let succ = g
            .succs
            .get(e.succ_ofs as usize..e.succ_ofs as usize + e.succ_len as usize)?;
        if succ.iter().any(|&c| c != e.inst) {
            return None;
        }
    }
    let list = |e: &StreamEnt| waits.get(e.wait_ofs as usize..e.wait_ofs as usize + e.wait_len as usize);
    let kept = |e: &StreamEnt, w: &Wait| {
        w.threshold != 0 && placement.get(w.id as usize).copied().flatten() == Some(e.seg)
    };
    let mut needed = vec![false; n];
    for e in g.gq_stream.iter().chain(&g.stream) {
        for w in list(e)? {
            if kept(e, w) {
                *needed.get_mut(w.id as usize)? = true;
            }
        }
    }
    let mut out = SegmentGates {
        stream: Vec::with_capacity(g.stream.len()),
        gq_stream: Vec::with_capacity(g.gq_stream.len()),
        waits: Vec::new(),
        succs: g.succs.clone(),
        dropped_waits: 0,
        dropped_succs: 0,
    };
    let mut lists = std::collections::HashMap::new();
    for (src, is_gq) in [(&g.stream, false), (&g.gq_stream, true)] {
        for e in src {
            let mut r = *e;
            let old = list(e)?;
            let (ofs, len) = *lists.entry((e.wait_ofs, e.wait_len, e.seg)).or_insert_with(|| {
                let ofs = out.waits.len() as u32;
                out.waits.extend(old.iter().filter(|w| kept(e, w)).copied());
                (ofs, out.waits.len() as u32 - ofs)
            });
            out.dropped_waits += e.wait_len as usize - len as usize;
            r.wait_ofs = ofs;
            r.wait_len = len as u16;
            if e.succ_len != 0 && !needed[e.inst as usize] {
                out.dropped_succs += e.succ_len as usize;
                r.succ_len = 0;
            }
            if is_gq {
                out.gq_stream.push(r);
            } else {
                out.stream.push(r);
            }
        }
    }
    if out.waits.is_empty() {
        out.waits.push(Wait::default());
    }
    Some(out)
}
