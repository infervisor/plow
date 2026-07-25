/-
# Plow.Memory — address-map reclamation safety.

Encodes a placed address map (mirroring `AddressMap` from `memory.rs`) carrying
the writer/reader task sets per buffer. States and proves:

* `mayReclaim` — the safety predicate from §6.2.2: two byte-overlapping
  buffers must have all readers of one happens-before all writers of the other.
* `AddressMapSound` — the address-map level invariant.
* `allocate_sound` — given the scheduler emits the bridging happens-before
  edges the allocator requires, the address map is sound.

The proof factors through `Plow.Protocol.protocol_covers_deps`: the bridging
edge is a data-dependency edge in the task graph, and the protocol covers it.
-/
import Plow.Basic
import Plow.Protocol

namespace Plow.Memory

open Plow Plow.Protocol

/-- One placed buffer, with the task sets that touch it. Mirrors
    `AddrEntry` but exposes the task-level info the safety proof needs. -/
structure AddrEntry (tg : TaskGraph) where
  name    : String
  offset  : Nat
  size    : Nat
  cls     : BufClass
  writers : List (Fin tg.n)
  readers : List (Fin tg.n)
  deriving Repr

/-- The reclamation predicate (§6.2.2): A and B may share bytes iff every
    reader of one happens-before every writer of the other. -/
def mayReclaim {tg : TaskGraph} (p : CounterProtocol tg)
    (a b : AddrEntry tg) : Prop :=
  (∀ r ∈ a.readers, ∀ w ∈ b.writers, happensBefore p r w) ∨
  (∀ r ∈ b.readers, ∀ w ∈ a.writers, happensBefore p r w)

/-- An address map is sound iff every pair of byte-overlapping, distinct-named
    entries satisfies `mayReclaim`. Distinct names matter because two entries
    with the same name are aliases (one zero-copy view of the other) and may
    share bytes by construction. -/
def AddressMapSound {tg : TaskGraph} (p : CounterProtocol tg)
    (entries : List (AddrEntry tg)) : Prop :=
  ∀ a b, a ∈ entries → b ∈ entries → a.name ≠ b.name →
    bytesOverlap a.offset a.size b.offset b.size →
    mayReclaim p a b

/-! ## Allocator obligations and soundness. -/

/-- The allocator's emitted obligations: for every co-located pair, a witness
    `(r, w)` such that the scheduler MUST have an edge `r → w` in `tg.edges`
    (so that `protocol_covers_deps` lifts it to `happensBefore`).

    Direction `.fwd` means readers-of-a happen-before writers-of-b (a is being
    reclaimed by b). `.bwd` is the symmetric case. -/
inductive ReclaimDir | fwd | bwd

structure ReclaimObligation (tg : TaskGraph) where
  a   : String  -- name of the older buffer (or symmetric)
  b   : String  -- name of the newer buffer
  dir : ReclaimDir

/-- The contract the allocator hands to the scheduler: for every overlapping
    pair, at least one obligation pins a bridging task pair *and* the bridging
    task pair, together with the universal reader/writer pinning, gives the
    full `mayReclaim` predicate.

    Concretely: the scheduler is required to emit, for the chosen `dir`, a
    counter-or-resource edge between every (reader of A, writer of B) pair.
    This is exactly what `tg.edges` records. -/
def ObligationsCoverOverlap {tg : TaskGraph} (p : CounterProtocol tg)
    (entries : List (AddrEntry tg))
    (obligations : List (ReclaimObligation tg)) : Prop :=
  ∀ a b, a ∈ entries → b ∈ entries → a.name ≠ b.name →
    bytesOverlap a.offset a.size b.offset b.size →
    ∃ o ∈ obligations,
      ((o.a = a.name ∧ o.b = b.name ∧ o.dir = ReclaimDir.fwd ∧
        ∀ r ∈ a.readers, ∀ w ∈ b.writers, happensBefore p r w) ∨
       (o.a = b.name ∧ o.b = a.name ∧ o.dir = ReclaimDir.fwd ∧
        ∀ r ∈ b.readers, ∀ w ∈ a.writers, happensBefore p r w))

/-! ## Main theorem (§6.2.2). -/

/-- Reclamation safety: if the obligations cover every byte overlap, the
    address map is sound. Direct consequence of unfolding the definitions.

    **Interface lemma.** `ObligationsCoverOverlap` already embeds the
    required `happensBefore` facts, so this theorem is definitional
    repackaging — it proves nothing about a concrete allocator run. The
    real trust story is `allocate_sound_from_edges` below (which derives
    the happens-before via `protocol_covers_deps` from plain graph edges)
    together with the executable per-schedule checker in
    `Plow/Verify.lean`, which validates those edge obligations. -/
theorem allocate_sound {tg : TaskGraph} (p : CounterProtocol tg)
    (entries : List (AddrEntry tg))
    (obligations : List (ReclaimObligation tg))
    (h_cover : ObligationsCoverOverlap p entries obligations) :
    AddressMapSound p entries := by
  intro a b ha hb hne hov
  obtain ⟨_, _, hcase⟩ := h_cover a b ha hb hne hov
  rcases hcase with ⟨_, _, _, hfwd⟩ | ⟨_, _, _, hbwd⟩
  · exact Or.inl hfwd
  · exact Or.inr hbwd

/-! ## Corollary: integration with `protocol_covers_deps`.

    If the scheduler does the natural thing (puts every required bridging
    edge into `tg.edges`), then `WellFormed`'s `edgeCovered` clause means the
    obligations are automatically satisfied. -/

/-- A simpler interface for the common case: state the obligations as a list
    of required `(reader, writer)` task pairs in the bridging direction, and
    require they appear in `tg.edges`. `protocol_covers_deps` then provides
    the `happensBefore`. -/
def EdgeBackedObligations {tg : TaskGraph} (_p : CounterProtocol tg)
    (entries : List (AddrEntry tg)) : Prop :=
  ∀ a b, a ∈ entries → b ∈ entries → a.name ≠ b.name →
    bytesOverlap a.offset a.size b.offset b.size →
    ((∀ r ∈ a.readers, ∀ w ∈ b.writers, (r, w) ∈ tg.edges) ∨
     (∀ r ∈ b.readers, ∀ w ∈ a.writers, (r, w) ∈ tg.edges))

/-- Edge-backed obligations + a well-formed protocol ⇒ a sound address map. -/
theorem allocate_sound_from_edges {tg : TaskGraph} (p : CounterProtocol tg)
    (wf : WellFormed p)
    (entries : List (AddrEntry tg))
    (h_edges : EdgeBackedObligations p entries) :
    AddressMapSound p entries := by
  intro a b ha hb hne hov
  rcases h_edges a b ha hb hne hov with hf | hb'
  · refine Or.inl ?_
    intro r hr w hw
    exact protocol_covers_deps p wf (r, w) (hf r hr w hw)
  · refine Or.inr ?_
    intro r hr w hw
    exact protocol_covers_deps p wf (r, w) (hb' r hr w hw)

end Plow.Memory
