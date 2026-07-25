//! §C Per-head KV growable pool — the runtime realization of
//! `lean-plow/Plow/KvPool.lean`.
//!
//! A model's attention layer owns one growable buffer; this type carves that
//! buffer into **per-(kv, head, seq) head-slots** so a FlashAttention kernel
//! reads one head's K (or V) for a sequence as a single contiguous DMA (rule
//! R-K1). The byte address of a head-slot is defined **identically** to the
//! Lean `headSlotOffset` so the R-K2 (byte-disjointness) and R-K4 (eviction)
//! proofs describe the code that actually runs — not a paraphrase of it.
//!
//! Layout (kv-major → head → seq; token positions grow inner-most *inside* a
//! slot and are therefore invisible to the offset formula):
//!
//! ```text
//! headSlotOffset(kv, head, seq)
//!   = base + ((kv·kvHeads + head)·maxSeqs + seq) · headSlotBytes
//! ```
//!
//! `headSlotBytes = max_seq_len · head_dim · elem_bytes` is the per-slot reserve;
//! `kvFactor` is 2 (separate K and V) or 1 (fused). See
//! `plow_asset::KvPaging` for the compiler-emitted geometry.

use plow_asset::KvPaging;

/// One attention layer's per-head growable pool. Mirrors Lean
/// `Plow.KvPool.GrowablePool` field-for-field (`base`, `kvFactor`, `kvHeads`,
/// `maxSeqs`, `headSlotBytes`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GrowablePool {
    /// Physical base address of this layer's buffer (block 0 of the pool).
    pub base: u64,
    /// Distinct runs per (head, seq): 2 for separate K and V, 1 if fused.
    pub kv_factor: u32,
    /// Number of distinct kv-heads.
    pub kv_heads: u32,
    /// Max sequences in flight the pool reserves head-slots for.
    pub max_seqs: u32,
    /// Reserve per `(kv, head, seq)` head-slot = `max_seq_len · head_dim · elem`.
    pub head_slot_bytes: u64,
}

impl GrowablePool {
    /// Build a layer's pool at physical `base` from the compiler-emitted
    /// `KvPaging` geometry. Negative geometry (shouldn't happen) clamps to 0.
    pub fn from_paging(base: u64, paging: &KvPaging) -> Self {
        GrowablePool {
            base,
            kv_factor: paging.kv_factor.max(0) as u32,
            kv_heads: paging.kv_heads.max(0) as u32,
            max_seqs: paging.max_seqs.max(0) as u32,
            head_slot_bytes: paging.head_slot_bytes,
        }
    }

    /// `true` iff `(kv, head, seq)` is a valid head-slot index for this pool.
    /// Mirrors Lean `Plow.KvPool.InRange`.
    #[inline]
    pub fn in_range(&self, kv: u32, head: u32, seq: u32) -> bool {
        kv < self.kv_factor && head < self.kv_heads && seq < self.max_seqs
    }

    /// Flattened slot index `(kv·kvHeads + head)·maxSeqs + seq` in `u64` (the
    /// order the Lean proof peels: seq against maxSeqs, then head against
    /// kvHeads). Widened to `u64` before every multiply so large pools don't
    /// overflow.
    #[inline]
    pub fn flat_index(&self, kv: u32, head: u32, seq: u32) -> u64 {
        (kv as u64 * self.kv_heads as u64 + head as u64) * self.max_seqs as u64
            + seq as u64
    }

    /// Byte offset of head-slot `(kv, head, seq)` — **byte-identical** to Lean
    /// `Plow.KvPool.headSlotOffset`. No bounds check (matches the total
    /// function in the proof); callers gate with [`GrowablePool::in_range`].
    #[inline]
    pub fn head_slot_offset(&self, kv: u32, head: u32, seq: u32) -> u64 {
        self.base + self.flat_index(kv, head, seq) * self.head_slot_bytes
    }

    /// Bounds-checked [`GrowablePool::head_slot_offset`]: `None` when
    /// `(kv, head, seq)` is out of range.
    #[inline]
    pub fn checked_offset(&self, kv: u32, head: u32, seq: u32) -> Option<u64> {
        self.in_range(kv, head, seq)
            .then(|| self.head_slot_offset(kv, head, seq))
    }

    /// Physical address of `position`'s bytes within head-slot `(kv, head, seq)`.
    /// Positions grow inner-most (R-K1 contiguity): `slot_base + position ·
    /// head_dim · elem`. `stride_bytes = head_dim · elem` is the per-token byte
    /// stride (`head_slot_bytes / max_seq_len`). `None` if the slot is out of
    /// range or the position spills past the slot's reserve.
    #[inline]
    pub fn token_addr(
        &self,
        kv: u32,
        head: u32,
        seq: u32,
        position: u64,
        stride_bytes: u64,
    ) -> Option<u64> {
        let base = self.checked_offset(kv, head, seq)?;
        let off = position.checked_mul(stride_bytes)?;
        (off + stride_bytes <= self.head_slot_bytes).then_some(base + off)
    }

    /// Total bytes the pool spans: `kvFactor · kvHeads · maxSeqs · headSlotBytes`.
    #[inline]
    pub fn pool_bytes(&self) -> u64 {
        self.kv_factor as u64
            * self.kv_heads as u64
            * self.max_seqs as u64
            * self.head_slot_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The exact Lean `headSlotOffset` formula, transcribed for the fidelity
    /// check. If `GrowablePool::head_slot_offset` ever diverges from this the
    /// `KvPool.lean` R-K2/R-K4 proofs stop describing the runtime (vacuous).
    fn lean_head_slot_offset(
        base: u64,
        kv_heads: u64,
        max_seqs: u64,
        head_slot_bytes: u64,
        kv: u64,
        head: u64,
        seq: u64,
    ) -> u64 {
        base + ((kv * kv_heads + head) * max_seqs + seq) * head_slot_bytes
    }

    /// Lean `bytesOverlap a la b lb` — half-open byte ranges intersect.
    fn bytes_overlap(a: u64, la: u64, b: u64, lb: u64) -> bool {
        a < b + lb && b < a + la
    }

    fn pool() -> GrowablePool {
        GrowablePool {
            base: 0x1_0000,
            kv_factor: 2,
            kv_heads: 5, // GQA-ish: non-power-of-two to exercise the mixed radix
            max_seqs: 3,
            head_slot_bytes: 256,
        }
    }

    #[test]
    fn offset_matches_lean_formula_over_the_domain() {
        let p = pool();
        for kv in 0..p.kv_factor {
            for head in 0..p.kv_heads {
                for seq in 0..p.max_seqs {
                    let got = p.head_slot_offset(kv, head, seq);
                    let want = lean_head_slot_offset(
                        p.base,
                        p.kv_heads as u64,
                        p.max_seqs as u64,
                        p.head_slot_bytes,
                        kv as u64,
                        head as u64,
                        seq as u64,
                    );
                    assert_eq!(got, want, "diverged at ({kv},{head},{seq})");
                }
            }
        }
    }

    #[test]
    fn distinct_head_slots_are_byte_disjoint() {
        // R-K2 mirror: every pair of distinct in-range head-slots occupies
        // disjoint [offset, offset + head_slot_bytes) byte ranges.
        let p = pool();
        let all: Vec<(u32, u32, u32)> = (0..p.kv_factor)
            .flat_map(|kv| {
                (0..p.kv_heads)
                    .flat_map(move |h| (0..p.max_seqs).map(move |s| (kv, h, s)))
            })
            .collect();
        for (i, &a) in all.iter().enumerate() {
            for &b in &all[i + 1..] {
                let oa = p.head_slot_offset(a.0, a.1, a.2);
                let ob = p.head_slot_offset(b.0, b.1, b.2);
                assert!(
                    !bytes_overlap(oa, p.head_slot_bytes, ob, p.head_slot_bytes),
                    "head-slots {a:?} and {b:?} overlap"
                );
            }
        }
        // And the whole grid packs into exactly `pool_bytes` past `base`.
        assert_eq!(
            p.pool_bytes(),
            p.kv_factor as u64 * p.kv_heads as u64 * p.max_seqs as u64 * p.head_slot_bytes
        );
    }

    #[test]
    fn in_range_and_checked_offset_gate_the_domain() {
        let p = pool();
        assert!(p.in_range(1, 4, 2));
        assert!(!p.in_range(2, 0, 0)); // kv == kv_factor
        assert!(!p.in_range(0, 5, 0)); // head == kv_heads
        assert!(!p.in_range(0, 0, 3)); // seq == max_seqs
        assert_eq!(p.checked_offset(2, 0, 0), None);
        assert_eq!(p.checked_offset(0, 0, 0), Some(p.base));
    }

    #[test]
    fn token_addr_is_contiguous_and_bounded() {
        // head_slot_bytes = 256, stride = head_dim*elem. Say head_dim*elem = 32
        // ⇒ 8 positions per slot.
        let p = pool();
        let stride = 32u64;
        let base = p.head_slot_offset(0, 0, 1);
        assert_eq!(p.token_addr(0, 0, 1, 0, stride), Some(base));
        assert_eq!(p.token_addr(0, 0, 1, 7, stride), Some(base + 7 * 32));
        // position 8 spills past the 256-byte slot.
        assert_eq!(p.token_addr(0, 0, 1, 8, stride), None);
        // out-of-range slot.
        assert_eq!(p.token_addr(2, 0, 0, 0, stride), None);
    }

    #[test]
    fn from_paging_reads_geometry() {
        let paging = KvPaging {
            block_tokens: 4,
            block_bytes: 64,
            kv_heads: 5,
            head_dim: 8,
            kv_factor: 2,
            max_seqs: 3,
            head_slot_bytes: 256,
            per_layer: vec![],
        };
        let p = GrowablePool::from_paging(0x2000, &paging);
        assert_eq!(
            p,
            GrowablePool {
                base: 0x2000,
                kv_factor: 2,
                kv_heads: 5,
                max_seqs: 3,
                head_slot_bytes: 256,
            }
        );
    }
}
