//! Boundary-tree retention policy (every pool, CUDA and AMD): every published boundary records its
//! canonical parent (the longest retained boundary below it on its chain) and the retained
//! boundaries that continue it; eviction is one LRU over unreserved entries, an attach refreshes
//! the attached boundary and its ancestors, and a boundary is retired structurally once exactly
//! one retained continuation extends it. No per-boundary flags, tiers or recency windows.
//!
//! Invariants (checked by `check_tree` after every step of the scripted sim cells):
//! * `users > 0` or an [`Await`] on a snapshot ⇒ it is never freed, on any path;
//! * `parent(C).rows < C.rows` and the parent lies on `C`'s chain (canonical when `C` was
//!   published: a shorter boundary published later does not adopt `C`); `None` iff none did;
//! * `children` hold retained ids only: every free unlinks synchronously
//!   ([`unlink_snapshot`]) and re-parents the freed boundary's children to its parent;
//! * a boundary is retired only when exactly one retained continuation extends it, every
//!   request whose prompt ended there has been continued (`ends`), it is not in use, not
//!   awaited, and not a shared-prefix end ([`Kind::SHARED_END`]: published because
//!   another prompt shares it, so its second continuation is expected, and it survives by
//!   LRU refresh alone).
use super::*;

/// Refresh `last_used` of `parent` and every ancestor above it (O(depth)).
pub(super) fn refresh_ancestors(inner: &mut Inner, mut parent: Option<u64>, tick: u64) {
    let mut hops = 0u32;
    while let Some(id) = parent {
        let Some(snap) = snap_mut(inner, id) else {
            debug_assert!(false, "parent link to a freed snapshot");
            break;
        };
        snap.last_used = tick;
        parent = snap.parent;
        hops += 1;
        if hops > 1 << 16 {
            tracing::error!("vmm prefix: boundary parent cycle");
            break;
        }
    }
}

/// The longest retained boundary strictly below `rows` on the chain of `tokens` whose radix
/// match is `m` (the walk `try_attach` does to pick an attach point).
pub(super) fn canonical_parent(
    inner: &Inner,
    m: &crate::memory::prefix::Match,
    tokens: &[u32],
    rows: u32,
    block_rows: u32,
) -> Option<u64> {
    let br = block_rows as usize;
    let mut best: Option<(u32, u64)> = None;
    for blocks in 0..=m.blocks {
        let start = blocks * br;
        if start >= rows as usize {
            break;
        }
        let node = blocks.checked_sub(1).map(|i| m.placed[i]);
        let Some(list) = inner.published.get(&node) else {
            continue;
        };
        for snap in list {
            if snap.rows < rows
                && tokens.get(start..snap.rows as usize) == Some(snap.tail.as_slice())
                && best.is_none_or(|(r, _)| snap.rows > r)
            {
                best = Some((snap.rows, snap.id));
            }
        }
    }
    best.map(|(_, id)| id)
}

/// Link the boundary `id` (just published or re-published) under its canonical `parent`.
pub(super) fn link(inner: &mut Inner, id: u64, parent: Option<u64>) {
    let Some(child) = snap_mut(inner, id) else {
        return;
    };
    let old = child.parent;
    if old != parent {
        child.parent = parent;
        if let Some(prev) = old.and_then(|p| snap_mut(inner, p)) {
            prev.children.retain(|&c| c != id);
        }
    }
    if let Some(p) = parent.and_then(|p| snap_mut(inner, p)) {
        if !p.children.contains(&id) {
            p.children.push(id);
        }
    }
}

/// Retire the parent of the retained boundary `id` if `id` is its only continuation.
pub(super) fn retire_parent_of(s: &Shared, inner: &mut Inner, id: u64) {
    if let Some(pid) = snap_ref(inner, id).and_then(|c| c.parent) {
        try_retire(s, inner, pid);
    }
}

/// Structural retire: free `id` when exactly one retained boundary continues it and nothing
/// holds, awaits or shares it. Its child re-parents to its parent, which is then checked by
/// the same rule ([`unlink_snapshot`]).
pub(super) fn try_retire(s: &Shared, inner: &mut Inner, id: u64) -> bool {
    let Some(p) = snap_ref(inner, id) else {
        return false;
    };
    if p.children.len() != 1
        || p.users != 0
        || p.pending != 0
        || p.ends != 0
        || p.kind.has(Kind::SHARED_END)
        || snap_awaited(&inner.awaited, p)
    {
        return false;
    }
    let bytes = p.bytes;
    if with_cause(inner, Some(Mechanism::Retire), None, |inner| {
        remove_snapshot_id(s, inner, id)
    }) {
        inner.stats.snapshots_retired += 1;
        tracing::debug!(id, bytes, "vmm prefix: boundary retired (superseded)");
        true
    } else {
        false
    }
}

/// `snap` left `published`: re-parent its children to its parent, drop it from the parent's
/// set, and apply the single-continuation rule to the parent.
pub(super) fn unlink_snapshot(s: &Shared, inner: &mut Inner, snap: &Snap) {
    if snap.parent.is_none() && snap.children.is_empty() {
        return;
    }
    let gp = snap.parent;
    for &c in &snap.children {
        if let Some(child) = snap_mut(inner, c) {
            child.parent = gp;
        }
        if let Some(g) = gp.and_then(|g| snap_mut(inner, g)) {
            if !g.children.contains(&c) {
                g.children.push(c);
            }
        }
    }
    let Some(g) = gp else {
        return;
    };
    if let Some(gs) = snap_mut(inner, g) {
        gs.children.retain(|&c| c != snap.id);
    }
    try_retire(s, inner, g);
}

/// One LRU: the least recently used snapshot nobody holds or awaits, with the blocks only it
/// kept attachable; then a dead radix leaf; then (not at the soft cap while leases are held)
/// the most recently used one. `false` = nothing evictable.
pub(super) fn evict_one_tree(s: &Shared, inner: &mut Inner, preserve_hot: bool) -> bool {
    // OOM reclamation needs device bytes now: a spill already in flight is the cheapest
    // source (the cap trim credits in-flight spills instead and never waits).
    if !preserve_hot && finish_oldest_spill(s, inner) {
        return true;
    }
    let awaited = inner.awaited.clone();
    // One LRU key; the newest boundary goes when it is the last candidate (the request that
    // published it is held by its lease, not by its snapshot). A session boundary inside its
    // TTL goes only once nothing unprotected is left, the host tier's entries included.
    // The soft-cap trim (`preserve_hot`) keeps the most recently used boundary while a live
    // request maps its blocks (production's MRU exemption, without its tier); allocation OOM
    // may take it.
    let hot = if preserve_hot && inner.stats.cache_blocks > 0 {
        inner
            .published
            .iter()
            .flat_map(|(node, list)| list.iter().map(move |snap| (*node, snap)))
            .max_by_key(|(_, snap)| snap.last_used)
            .filter(|(node, _)| {
                node.is_some_and(|k| inner.cache.node_refs(k).is_some_and(|r| r > 0))
            })
            .map(|(_, snap)| snap.id)
    } else {
        None
    };
    let now = std::time::Instant::now();
    let lru = |protected: bool| {
        inner
            .published
            .values()
            .flatten()
            .filter(|snap| snap.users == 0 && snap.on_device() && !snap_awaited(&awaited, snap))
            .filter(|snap| Some(snap.id) != hot)
            .filter(|snap| snap.protect_until.is_some_and(|t| t > now) == protected)
            .min_by_key(|snap| snap.last_used)
            .map(|snap| snap.id)
    };
    let victim = lru(false);
    let protected = lru(true);
    // Blocks no boundary makes attachable are the cheapest bytes.
    if evict_dead_leaf(s, inner) {
        return true;
    }
    // A cold host boundary (published on a first sighting, never attached) ranks below every
    // used device boundary: it goes before the device victim is spilled or freed.
    if victim.is_some_and(|id| !snap_ref(inner, id).is_some_and(Snap::cold)) {
        if let Some(id) = host_lru(inner).filter(|&id| snap_ref(inner, id).is_some_and(Snap::cold))
        {
            with_cause(inner, Some(Mechanism::Cold), None, |inner| {
                evict_unit(s, inner, id)
            });
            return true;
        }
    }
    if let Some(id) = victim {
        // Level 2 of the same LRU: the device victim moves to pinned host memory when a tier
        // is armed; its blocks stay (a host snapshot is attachable through them).
        if spill_unit(s, inner, id) {
            return true;
        }
        evict_unit(s, inner, id);
        return true;
    }
    // Every unprotected device boundary is held or awaited: the device bytes a host-resident
    // boundary still occupies (its blocks) are the next level of the same LRU.
    if let Some(id) = host_lru(inner) {
        evict_unit(s, inner, id);
        return true;
    }
    if let Some(id) = protected {
        if spill_unit(s, inner, id) {
            return true;
        }
        evict_unit(s, inner, id);
        return true;
    }
    // Whole LRU radix leaves, never through a held, awaited or in-flight boundary.
    let Inner {
        cache,
        published,
        awaited,
        ..
    } = &mut *inner;
    let Some(key) = cache.evict_lru_where(|key| {
        published
            .get(&Some(key))
            .is_none_or(|list| snapshots_free(list, awaited))
    }) else {
        return false;
    };
    cache_changed(s);
    if let Some(ids) = inner.node_blocks.remove(&key) {
        inner.stats.cache_blocks -= ids.len() as u64;
        inner.stats.cache_bytes -= ids.len() as u64 * s.block_bytes;
        for id in ids {
            deref_block(s, inner, id);
        }
    }
    with_cause(inner, Some(Mechanism::NodeLru), None, |inner| {
        drop_node_snapshots(s, inner, key)
    });
    inner.stats.nodes_evicted += 1;
    true
}

/// Evict the boundary `id` and every radix leaf it alone kept attachable.
pub(super) fn evict_unit(s: &Shared, inner: &mut Inner, id: u64) {
    let before = inner.stats.cache_bytes;
    let mech = (inner.drop_mech == Mechanism::Other).then_some(Mechanism::Lru);
    if !with_cause(inner, mech, None, |inner| remove_snapshot_id(s, inner, id)) {
        return;
    }
    let mut blocks = 0;
    while evict_dead_leaf(s, inner) {
        blocks += 1;
    }
    inner.stats.eviction_units += 1;
    inner.stats.eviction_unit_bytes += before - inner.stats.cache_bytes;
    tracing::debug!(
        freed = before - inner.stats.cache_bytes,
        blocks,
        "vmm prefix eviction unit"
    );
}

/// Test hook: the boundary tree's structural invariants over every published snapshot.
#[cfg(test)]
pub(super) fn check_tree(inner: &Inner) {
    use std::collections::HashMap;
    let snaps: HashMap<u64, &Snap> = inner
        .published
        .values()
        .flatten()
        .map(|s| (s.id, s))
        .collect();
    assert_eq!(
        snaps.len(),
        inner.by_id.len(),
        "by_id out of step with published"
    );
    for (&id, snap) in &snaps {
        assert_eq!(inner.by_id.get(&id), Some(&inner.by_id[&id]));
        if let Some(p) = snap.parent {
            let parent = snaps
                .get(&p)
                .unwrap_or_else(|| panic!("snap {id}: parent {p} freed"));
            assert!(parent.rows < snap.rows, "snap {id}: parent not below it");
            assert!(
                parent.children.contains(&id),
                "snap {id}: not in its parent's children"
            );
        }
        for &c in &snap.children {
            let child = snaps
                .get(&c)
                .unwrap_or_else(|| panic!("snap {id}: child {c} freed"));
            assert_eq!(child.parent, Some(id), "child {c}: parent link mismatch");
        }
        assert!(
            snap.children.len() != 1
                || snap.users > 0
                || snap.pending > 0
                || snap.ends > 0
                || snap.kind.has(Kind::SHARED_END)
                || snap_awaited(&inner.awaited, snap),
            "snap {id}: single retained continuation left unretired"
        );
    }
}
