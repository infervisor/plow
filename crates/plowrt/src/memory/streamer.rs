//! §N Weights/KV tiered streamer — offload, prefetch, HBM reclaim.
//!
//! Streams weights and KV blocks between HBM (hot) ↔ host-pinned (warm) ↔
//! NVMe/DPU (cold) so the resident HBM working set is bounded and elastic. This
//! is the mechanism behind the §C/§M pressure paths (FlexGen / ZeRO-Inference
//! layer-wise offload + paged-KV offload), driven by counters so streaming
//! overlaps compute.
//!
//! Skeleton: the residency table + pressure controller policy. Actual transfers
//! delegate to the §M DMA plane; the CPU backend models every tier as memcpy so
//! eviction/prefetch/pressure logic is testable without a GPU.
//!
//! **[`KvArena`]** is the per-model owner of KV allocation. Each attention
//! layer has its own `BlockAllocator` (compiler emits one `kv_cache_L{i}`
//! growable buffer per layer, all heads packed inside a block). The mux calls
//! [`KvArena::allocate_slot`] on admission — one `PageTable` per layer sized
//! for the slot's `prompt_len + max_tokens` — and [`KvArena::release_slot`]
//! on slot free. OOM is a first-class result: the mux sheds the request.

/// Where a weight layer-block or KV block currently resides.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tier {
    /// Resident in device HBM.
    Hbm,
    /// Evicted to host-pinned RAM.
    HostPinned,
    /// Evicted to NVMe / DPU storage.
    Cold,
}

/// Per-model weight residency mode, chosen at load by HBM budget.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightMode {
    /// Whole model resident (fits HBM) — loaded once.
    Resident,
    /// Sliding window of `window` layer-blocks kept in HBM; the rest streamed.
    Streamed { window: u32 },
}

/// Reclaim actions, in the priority order the pressure controller applies them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reclaim {
    /// Evict cold KV blocks to a lower tier.
    EvictKv,
    /// Flip a low-traffic model from Resident → Streamed weights.
    StreamWeights,
    /// Preempt a low-priority sequence.
    Preempt,
}

/// Per-model HBM reservation handle. Drop releases the budget.
pub struct Reservation {
    pub slug: String,
    pub bytes: u64,
}

/// HBM watermark + reclaim policy + per-model budget tracking.
pub struct Streamer {
    capacity: u64,
    resident: u64,
    /// Per-model reserved budgets.
    reservations: Vec<Reservation>,
    /// Fraction of capacity above which reclaim triggers.
    high_watermark: f64,
}

impl Streamer {
    pub fn new(capacity: u64) -> Self {
        Streamer {
            capacity,
            resident: 0,
            reservations: Vec::new(),
            high_watermark: 0.9,
        }
    }

    /// Reserve `bytes` of HBM budget for a model identified by `slug`.
    /// Returns `Err` if the reservation would exceed capacity.
    pub fn reserve(&mut self, slug: &str, bytes: u64) -> crate::Result<()> {
        let total_reserved: u64 = self.reservations.iter().map(|r| r.bytes).sum();
        if total_reserved + bytes > self.capacity {
            return Err(crate::RuntimeError::Oom(format!(
                "HBM budget: {slug} needs {bytes}B, only {}B available",
                self.capacity.saturating_sub(total_reserved)
            )));
        }
        self.reservations.push(Reservation {
            slug: slug.to_string(),
            bytes,
        });
        Ok(())
    }

    /// Release the budget reservation for a model.
    pub fn release_reservation(&mut self, slug: &str) {
        self.reservations.retain(|r| r.slug != slug);
    }

    /// Query remaining unreserved HBM.
    pub fn available(&self) -> u64 {
        let reserved: u64 = self.reservations.iter().map(|r| r.bytes).sum();
        self.capacity.saturating_sub(reserved)
    }

    pub fn account(&mut self, delta: i64) {
        self.resident = (self.resident as i64 + delta).max(0) as u64;
    }

    pub fn utilization(&self) -> f64 {
        if self.capacity == 0 {
            return 0.0;
        }
        self.resident as f64 / self.capacity as f64
    }

    pub fn under_pressure(&self) -> bool {
        self.utilization() > self.high_watermark
    }

    /// The reclaim plan to satisfy `need` bytes (or relieve pressure). Ordered:
    /// evict cold KV, then stream weights, then preempt.
    pub fn reclaim_plan(&self, need: u64) -> Vec<Reclaim> {
        if !self.under_pressure() && self.resident + need <= self.capacity {
            return Vec::new();
        }
        vec![Reclaim::EvictKv, Reclaim::StreamWeights, Reclaim::Preempt]
    }

    /// Execute a reclaim plan, returning the total bytes freed. Each variant
    /// dispatches to the appropriate subsystem:
    /// - `EvictKv`: signals the KV arena to evict cold (LRU) blocks.
    /// - `StreamWeights`: flips a model from Resident → Streamed, freeing HBM.
    /// - `Preempt`: signals the mux to preempt the lowest-priority sequence.
    ///
    /// The actual freeing is a no-op until the backend is wired; this method
    /// returns a conservative estimate so the caller can retry allocation.
    pub fn execute_reclaim(&mut self, plan: &[Reclaim]) -> u64 {
        let freed = 0u64;
        for action in plan {
            match action {
                Reclaim::EvictKv => {
                    // Placeholder: in production this calls into the KV arena's
                    // LRU eviction path and frees head-slot pages.
                    // For now, account a symbolic amount to unblock the retry.
                }
                Reclaim::StreamWeights => {
                    // Placeholder: flip a cold model's weight mode to Streamed
                    // and account the freed resident bytes.
                }
                Reclaim::Preempt => {
                    // Placeholder: signal the mux to preempt a low-priority
                    // sequence. The actual slot release happens asynchronously.
                }
            }
            let _ = action; // suppress unused warning for placeholders
        }
        // In production, each arm would `self.account(-delta)` and add to freed.
        // The caller uses `freed > 0` to decide whether to retry.
        freed
    }

    /// Bounded HBM footprint of a streamed model: `window × per-block bytes`.
    pub fn streamed_footprint(mode: WeightMode, per_block_bytes: u64) -> u64 {
        match mode {
            WeightMode::Resident => u64::MAX,
            WeightMode::Streamed { window } => window as u64 * per_block_bytes,
        }
    }
}

use plow_asset::KvPaging;
use rustc_hash::FxHashMap;

use crate::memory::pool::GrowablePool;

/// Opaque handle the mux hands back to the arena on `release_slot`. Cloneable
/// only for debug/testing — the intent is one-owner-per-slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SlotHandle(pub u64);

/// OOM verdict when the arena can't satisfy `allocate_slot` — no free seq-slot
/// (`needed` is always 1; `available` is the free-slot count at the time).
#[derive(Debug)]
pub struct KvOom {
    pub needed: u32,
    pub available: u32,
}

impl std::fmt::Display for KvOom {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "kv oom: needed {} seq-slot, have {}", self.needed, self.available)
    }
}
impl std::error::Error for KvOom {}

/// Per-model **per-head** KV allocator (the `GrowablePool` model, §6.2.6 /
/// `lean-plow/Plow/KvPool.lean`).
///
/// Each attention layer owns one growable `kv_cache_L{i}` buffer carved into a
/// grid of `kv_factor × kv_heads × max_seqs` contiguous head-slots (see
/// [`GrowablePool`]). Admission hands each live request a **stable seq-slot**
/// `s ∈ [0, max_seqs)` held for the sequence's lifetime; the head-slot base for
/// `(layer, kv, head)` of that sequence is `pool[layer].head_slot_offset(kv,
/// head, s)`. The mux writes the per-`(row, layer)` seq-slot base into the
/// indirection table and the attention kernel strides by `(kv, head)` using the
/// pool geometry.
///
/// This replaces the earlier packed-block model (`memory::kv::BlockAllocator` /
/// `PageTable`), which is retained but unused pending a page-table-over-pool
/// view for prefix sharing.
pub struct KvArena {
    /// One growable pool per attention layer, in `paging.per_layer` order.
    pools: Vec<GrowablePool>,
    /// Max concurrent sequences (seq-slots) the pools reserve.
    max_seqs: u32,
    /// LIFO free list of seq-slot ids in `[0, max_seqs)` — hot slots stay warm.
    free_seqs: Vec<u32>,
    /// Live slots: handle → the seq-slot it holds.
    slots: FxHashMap<SlotHandle, u32>,
    next_handle: u64,
}

impl KvArena {
    /// Build the arena from the manifest's KV paging + a per-layer physical
    /// base address (from the `AddressSpace`). Layers without a base default to
    /// zero (addresses aren't meaningful until wired).
    ///
    /// When the manifest carries per-head geometry (`head_slot_bytes > 0`) the
    /// pools use it directly. Legacy maps (no geometry) derive an equivalent
    /// single-sequence grid from `block_bytes × initial_blocks` so a pool spans
    /// exactly the same reserved region the packed-block model did.
    pub fn new(paging: KvPaging, bases: &[u64]) -> Self {
        let kv_factor = if paging.kv_factor > 0 { paging.kv_factor as u32 } else { 2 };
        let kv_heads = paging.kv_heads.max(0) as u32;
        let kh = kv_heads.max(1); // divisor guard for the legacy derivation
        let max_seqs = if paging.max_seqs > 0 { paging.max_seqs as u32 } else { 1 };
        let pools = paging
            .per_layer
            .iter()
            .enumerate()
            .map(|(i, lp)| {
                let base = bases.get(i).copied().unwrap_or(0);
                let head_slot_bytes = if paging.head_slot_bytes > 0 {
                    paging.head_slot_bytes
                } else {
                    let reserved =
                        paging.block_bytes.saturating_mul(lp.initial_blocks.max(0) as u64);
                    let cells = kv_factor as u64 * kh as u64 * max_seqs as u64;
                    if cells == 0 { 0 } else { reserved / cells }
                };
                GrowablePool { base, kv_factor, kv_heads, max_seqs, head_slot_bytes }
            })
            .collect();
        KvArena {
            pools,
            max_seqs,
            free_seqs: (0..max_seqs).rev().collect(),
            slots: FxHashMap::default(),
            next_handle: 1,
        }
    }

    /// Free seq-slots remaining — the ceiling on new concurrent sequences.
    pub fn min_available(&self) -> u32 {
        self.free_seqs.len() as u32
    }

    /// Claim one stable seq-slot for a new sequence. `_seq_upper` is accepted
    /// for API symmetry (and a future R-K6 per-sequence-length check); the
    /// head-slot is pre-sized by the compiler for the bucket. `Err` when every
    /// seq-slot is in use — the mux sheds the request.
    pub fn allocate_slot(&mut self, _seq_upper: i64) -> Result<SlotHandle, KvOom> {
        let Some(seq) = self.free_seqs.pop() else {
            return Err(KvOom { needed: 1, available: 0 });
        };
        let handle = SlotHandle(self.next_handle);
        self.next_handle += 1;
        self.slots.insert(handle, seq);
        Ok(handle)
    }

    /// Return the slot's seq-slot to the free list. No-op for an unknown handle.
    pub fn release_slot(&mut self, handle: SlotHandle) {
        if let Some(seq) = self.slots.remove(&handle) {
            self.free_seqs.push(seq);
        }
    }

    /// Number of live slots.
    pub fn live_slots(&self) -> usize {
        self.slots.len()
    }

    /// The stable seq-slot id this handle holds, or `None` if stale.
    pub fn seq_slot(&self, handle: SlotHandle) -> Option<u32> {
        self.slots.get(&handle).copied()
    }

    /// Per-`(row, layer)` indirection base for this slot: the address of its
    /// `(kv=0, head=0, seq)` head-slot = `pool.base + seq × head_slot_bytes`.
    /// The attention kernel adds `(kv·kv_heads + head)·max_seqs·head_slot_bytes`
    /// (the separable tail of `headSlotOffset`) using the pool geometry. `None`
    /// for a stale handle or out-of-range layer.
    pub fn seq_slot_base(&self, handle: SlotHandle, layer: usize) -> Option<u64> {
        let seq = *self.slots.get(&handle)?;
        let pool = self.pools.get(layer)?;
        Some(pool.head_slot_offset(0, 0, seq))
    }

    /// Full per-head address of `(layer, kv, head)` for this slot's sequence —
    /// the byte-exact `headSlotOffset`. Used by tests / a host-side kernel.
    pub fn head_slot_addr(
        &self,
        handle: SlotHandle,
        layer: usize,
        kv: u32,
        head: u32,
    ) -> Option<u64> {
        let seq = *self.slots.get(&handle)?;
        let pool = self.pools.get(layer)?;
        pool.checked_offset(kv, head, seq)
    }

    /// The pool for `layer` (its geometry / base), or `None` if out of range.
    pub fn pool(&self, layer: usize) -> Option<&GrowablePool> {
        self.pools.get(layer)
    }

    /// Number of KV attention layers this arena tracks — matches the compiler's
    /// `KvPaging::per_layer.len()`.
    pub fn n_layers(&self) -> usize {
        self.pools.len()
    }

    /// Max concurrent sequences (seq-slots) the pools reserve.
    pub fn max_seqs(&self) -> u32 {
        self.max_seqs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use plow_asset::KvLayerPaging;

    /// Two-layer paging with explicit per-head geometry: `max_seqs` seq-slots,
    /// `kv_factor=2`, `kv_heads=2`, `head_slot_bytes=64`.
    fn paging_2layers(max_seqs: i64) -> KvPaging {
        KvPaging {
            block_tokens: 4,
            block_bytes: 64,
            kv_heads: 2,
            head_dim: 8,
            kv_factor: 2,
            max_seqs,
            head_slot_bytes: 64,
            per_layer: vec![
                KvLayerPaging {
                    layer_idx: 0,
                    buffer_name: "kv_cache_L0".into(),
                    initial_blocks: 4,
                },
                KvLayerPaging {
                    layer_idx: 1,
                    buffer_name: "kv_cache_L1".into(),
                    initial_blocks: 4,
                },
            ],
        }
    }

    #[test]
    fn allocate_and_release_roundtrip() {
        // 4 seq-slots across 2 layers.
        let mut a = KvArena::new(paging_2layers(4), &[0, 0]);
        assert_eq!(a.live_slots(), 0);
        assert_eq!(a.min_available(), 4);
        assert_eq!(a.n_layers(), 2);
        assert_eq!(a.max_seqs(), 4);

        let h = a.allocate_slot(8).unwrap();
        assert_eq!(a.live_slots(), 1);
        assert_eq!(a.min_available(), 3);
        assert!(a.seq_slot(h).is_some());

        a.release_slot(h);
        assert_eq!(a.live_slots(), 0);
        assert_eq!(a.min_available(), 4);
    }

    #[test]
    fn seq_slot_base_and_head_slot_addr_follow_the_pool_formula() {
        // Bases: layer 0 → 0x1000, layer 1 → 0x2000; head_slot_bytes = 64.
        let mut a = KvArena::new(paging_2layers(4), &[0x1000, 0x2000]);
        // LIFO free list (0..4).rev() ⇒ first pop is seq-slot 0.
        let h0 = a.allocate_slot(8).unwrap();
        let h1 = a.allocate_slot(8).unwrap();
        assert_eq!(a.seq_slot(h0), Some(0));
        assert_eq!(a.seq_slot(h1), Some(1));

        // Per-(row,layer) base = pool.base + seq × head_slot_bytes.
        assert_eq!(a.seq_slot_base(h0, 0), Some(0x1000));
        assert_eq!(a.seq_slot_base(h1, 0), Some(0x1000 + 64)); // seq 1
        assert_eq!(a.seq_slot_base(h0, 1), Some(0x2000)); // layer 1 base

        // Full head-slot address = base + ((kv·kv_heads+head)·max_seqs+seq)·hsb.
        // Layer 0, kv=1, head=0, seq=0: (1·2+0)·4 + 0 = 8 ⇒ 0x1000 + 8·64.
        assert_eq!(a.head_slot_addr(h0, 0, 1, 0), Some(0x1000 + 8 * 64));
        // Out-of-range head / layer / stale handle.
        assert_eq!(a.head_slot_addr(h0, 0, 0, 9), None);
        assert_eq!(a.seq_slot_base(h0, 5), None);
        a.release_slot(h0);
        assert_eq!(a.seq_slot_base(h0, 0), None);
    }

    #[test]
    fn oom_when_seq_slots_exhausted() {
        // Only 1 seq-slot: the second admission OOMs.
        let mut a = KvArena::new(paging_2layers(1), &[0, 0]);
        let _h = a.allocate_slot(8).unwrap();
        let err = a.allocate_slot(8).unwrap_err();
        assert_eq!(err.needed, 1);
        assert_eq!(err.available, 0);
        assert_eq!(a.live_slots(), 1);
    }

    #[test]
    fn legacy_geometry_derives_equivalent_single_seq_pool() {
        // No per-head geometry (head_slot_bytes = 0, max_seqs = 0): fall back to
        // one seq-slot spanning block_bytes(64) × initial_blocks(4) = 256 bytes,
        // split kv_factor(2) × kv_heads(2) = 4 head-slots of 64 bytes each.
        let mut paging = paging_2layers(0);
        paging.max_seqs = 0;
        paging.head_slot_bytes = 0;
        let a_arena = KvArena::new(paging, &[0x1000, 0x2000]);
        assert_eq!(a_arena.max_seqs(), 1);
        let pool = a_arena.pool(0).unwrap();
        assert_eq!(pool.head_slot_bytes, 64);
        assert_eq!(pool.pool_bytes(), 256); // == 64 × 4 reserved region
    }
}
