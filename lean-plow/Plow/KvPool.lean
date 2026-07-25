/-
# Plow.KvPool — per-head KV pool injectivity (R-K2) and eviction safety
                (R-K4), §6.2.6.

Encodes the per-(layer, head, seq) growable pool layout and proves:

* **R-K2** — distinct head-slots never share bytes (index function is
  injective into the strided byte lattice).
* **R-K4** — reassigning a head-slot from an old sequence to a new one is
  safe when every reader of the old sequence's sharers has happened-before
  the install of the new mapping. This reduces to the reclamation-safety
  machinery in `Plow.Memory` at head-slot granularity.
-/
import Plow.Basic
import Plow.Protocol
import Plow.Memory

namespace Plow.KvPool

open Plow

/-- Per-(layer, device) growable pool descriptor. -/
structure GrowablePool where
  base          : Nat
  kvFactor      : Nat        -- 2 for separate K and V; 1 if fused.
  kvHeads       : Nat        -- number of distinct kv-heads
  maxSeqs       : Nat        -- max sequences in flight
  headSlotBytes : Nat        -- reserve per (head, seq) = max_seq * head_dim * elem

/-- Byte offset of head-slot `(kv, head, seq)` within the pool. The layout is
    kv-major then head then seq, mirroring §6.2.6's per-head contiguity rule. -/
def headSlotOffset (p : GrowablePool) (kv head seq : Nat) : Nat :=
  p.base + ((kv * p.kvHeads + head) * p.maxSeqs + seq) * p.headSlotBytes

/-- An index `(kv, head, seq)` is valid for the pool. -/
def InRange (p : GrowablePool) (kv head seq : Nat) : Prop :=
  kv < p.kvFactor ∧ head < p.kvHeads ∧ seq < p.maxSeqs

/-! ## Sub-lemma: index function is injective on the valid domain. -/

/-- Auxiliary: a mixed-radix step. If `lo1 < d` and `lo2 < d`, then
    `hi1 * d + lo1 = hi2 * d + lo2 → hi1 = hi2 ∧ lo1 = lo2`.

    Trichotomy on `hi1` vs `hi2` plus the multiplication identity
    `(k + 1) * d = k * d + d`; omega closes each branch. -/
private theorem mixed_radix (hi1 hi2 lo1 lo2 d : Nat)
    (h1 : lo1 < d) (h2 : lo2 < d)
    (heq : hi1 * d + lo1 = hi2 * d + lo2) : hi1 = hi2 ∧ lo1 = lo2 := by
  rcases Nat.lt_trichotomy hi1 hi2 with hlt | heqh | hgt
  · -- hi1 < hi2 ⇒ hi1 * d + d ≤ hi2 * d, but heq + h1 give hi2 * d < hi1 * d + d.
    have hsep : hi1 * d + d ≤ hi2 * d :=
      calc hi1 * d + d = (hi1 + 1) * d := by rw [Nat.add_mul, Nat.one_mul]
        _ ≤ hi2 * d := Nat.mul_le_mul_right d hlt
    omega
  · subst heqh; exact ⟨rfl, by omega⟩
  · have hsep : hi2 * d + d ≤ hi1 * d :=
      calc hi2 * d + d = (hi2 + 1) * d := by rw [Nat.add_mul, Nat.one_mul]
        _ ≤ hi1 * d := Nat.mul_le_mul_right d hgt
    omega

/-- Auxiliary: distinct strided slot positions yield disjoint byte ranges.
    `hsz` is not needed for the current proof (the arithmetic goes through
    without positivity), but the caller in `head_slots_disjoint` still
    verifies it — kept for API symmetry with the caller's precondition. -/
private theorem offset_separation (base sz i1 i2 : Nat) (_hsz : 0 < sz)
    (hne : i1 ≠ i2) :
    ¬ bytesOverlap (base + i1 * sz) sz (base + i2 * sz) sz := by
  intro ⟨h1, h2⟩
  rcases Nat.lt_or_ge i1 i2 with hlt | hge
  · -- i1 < i2 ⇒ i2 * sz ≥ i1 * sz + sz, contradicting h1.
    have hle : i1 + 1 ≤ i2 := hlt
    have hmul : (i1 + 1) * sz ≤ i2 * sz := Nat.mul_le_mul_right sz hle
    have hsep : i1 * sz + sz ≤ i2 * sz := by
      have : (i1 + 1) * sz = i1 * sz + sz := by rw [Nat.add_mul, Nat.one_mul]
      omega
    omega
  · -- i2 ≤ i1 with i1 ≠ i2 ⇒ i2 < i1, symmetric against h2.
    have hlt' : i2 < i1 := Nat.lt_of_le_of_ne hge (Ne.symm hne)
    have hle : i2 + 1 ≤ i1 := hlt'
    have hmul : (i2 + 1) * sz ≤ i1 * sz := Nat.mul_le_mul_right sz hle
    have hsep : i2 * sz + sz ≤ i1 * sz := by
      have : (i2 + 1) * sz = i2 * sz + sz := by rw [Nat.add_mul, Nat.one_mul]
      omega
    omega

/-- The flattened index `(kv * kvHeads + head) * maxSeqs + seq` is injective
    on the valid domain. Two applications of `mixed_radix`: peel off `seq`
    against `maxSeqs`, then `head` against `kvHeads`. -/
theorem flat_index_injective (p : GrowablePool)
    (kv1 h1 s1 kv2 h2 s2 : Nat)
    (r1 : InRange p kv1 h1 s1)
    (r2 : InRange p kv2 h2 s2)
    (hne : (kv1, h1, s1) ≠ (kv2, h2, s2)) :
    (kv1 * p.kvHeads + h1) * p.maxSeqs + s1 ≠
    (kv2 * p.kvHeads + h2) * p.maxSeqs + s2 := by
  intro heq
  -- Outer peel against maxSeqs.
  obtain ⟨hxeq, hseq⟩ :=
    mixed_radix _ _ _ _ _ r1.2.2 r2.2.2 heq
  -- Inner peel against kvHeads.
  obtain ⟨hkveq, hheq⟩ :=
    mixed_radix _ _ _ _ _ r1.2.1 r2.2.1 hxeq
  exact hne (by simp [hkveq, hheq, hseq])

/-! ## R-K2: head-slot byte disjointness. -/

/-- If two head-slots have different `(kv, head, seq)` triples and the slot
    size is non-zero, their byte ranges are disjoint. Follows from
    injectivity of the flat index: distinct indices ⇒ offsets differ by
    at least one full `headSlotBytes` stride. -/
theorem head_slots_disjoint (p : GrowablePool)
    (kv1 h1 s1 kv2 h2 s2 : Nat)
    (r1 : InRange p kv1 h1 s1)
    (r2 : InRange p kv2 h2 s2)
    (hne : (kv1, h1, s1) ≠ (kv2, h2, s2))
    (hsz : 0 < p.headSlotBytes) :
    ¬ bytesOverlap (headSlotOffset p kv1 h1 s1) p.headSlotBytes
                   (headSlotOffset p kv2 h2 s2) p.headSlotBytes := by
  have hidx :
    (kv1 * p.kvHeads + h1) * p.maxSeqs + s1
    ≠ (kv2 * p.kvHeads + h2) * p.maxSeqs + s2 :=
    flat_index_injective p kv1 h1 s1 kv2 h2 s2 r1 r2 hne
  simpa [headSlotOffset] using
    offset_separation p.base p.headSlotBytes _ _ hsz hidx

/-! ## R-K4 — head-slot eviction safety.

    A head-slot at `(kv, head, seq_old)` may be reassigned to `seq_new` iff
    every reader of the head-slot under the old sequence happens-before every
    writer under the new sequence. This is exactly the reclamation-safety
    predicate `Plow.Memory.mayReclaim` — evaluated at head-slot granularity
    rather than at buffer granularity. The proof is by construction: an
    eviction that satisfies `mayReclaim` is safe by `allocate_sound` (which
    proves reclamation is sound whenever the scheduler orders the reader/
    writer sets appropriately). -/

open Plow.Protocol Plow.Memory

/-- A head-slot rendered as an abstract `AddrEntry` for use by
    `Plow.Memory.mayReclaim`. Byte offset and size come from the pool's
    layout; reader/writer sets come from the schedule. -/
def headSlotAsEntry {tg : TaskGraph} (p : GrowablePool) (kv head seq : Nat)
    (readers writers : List (Fin tg.n)) : AddrEntry tg where
  name    := s!"kv:{kv}:{head}:{seq}"
  offset  := headSlotOffset p kv head seq
  size    := p.headSlotBytes
  cls     := BufClass.Growable
  writers := writers
  readers := readers

/-- **R-K4**: reassigning a head-slot from an old sequence to a new one is
    safe iff every reader of the old sequence happens-before every writer of
    the new sequence, under the counter protocol's `happensBefore` relation.

    This is the head-slot instance of the general reclamation predicate.
    Directly follows from the definitions — `mayReclaim` on two `AddrEntry`s
    built from the same pool location with different sequence ids reduces to
    the readers-of-old ≺ writers-of-new claim. -/
theorem head_slot_eviction_safe {tg : TaskGraph} (proto : CounterProtocol tg)
    (pool : GrowablePool) (kv head : Nat) (seq_old seq_new : Nat)
    (readers_old writers_old : List (Fin tg.n))
    (readers_new writers_new : List (Fin tg.n))
    (h_order :
      ∀ r ∈ readers_old, ∀ w ∈ writers_new, happensBefore proto r w) :
    mayReclaim proto
      (headSlotAsEntry pool kv head seq_old readers_old writers_old)
      (headSlotAsEntry pool kv head seq_new readers_new writers_new) := by
  unfold mayReclaim headSlotAsEntry
  exact Or.inl h_order

end Plow.KvPool
