/-
# Plow.TransitiveReduction — soundness of dropping implied counter edges.

The emitter (`packet::devbuild::Builder::finish`) removes two kinds of counter
wait before handing the protocol to the runtime:

  1. a COARSE dep `A→C` whose ordering a path `A→…→C` already provides
     (transitive reduction), and
  2. a REPEATED wait on the same counter within one task's wait list.

Measured on the 93-layer Kimi-K3 decode program: 69 implied edges and 138
duplicate waits, together 52,992 of 454,942 runtime polls (11.6%).

## What has to be proven, and why it is not `protocol_covers_deps`

`Plow.Protocol.WellFormed.edgeCovered` demands that every data edge be
*directly* `counterGated ∨ resourceOrdered`. Removal (1) deliberately breaks
that: after it, the edge `A→C` is ordered only through `A→B→C`. So the
reduction is NOT sound with respect to `edgeCovered`, and claiming it under
`protocol_covers_deps` would be false.

What survives — and what the runtime actually needs — is coverage in terms of
`happensBefore`, which is transitively closed by construction. This file states
that weaker invariant (`CoversHB`), shows it is what `WellFormed` already
implies, and proves both removals preserve it.

Removal (2) is stronger: it preserves `counterGated` itself, hence `edgeCovered`
too, because `counterGated` only asks for membership and a duplicate entry adds
no membership. It is proven separately for that reason.
-/
import Plow.Protocol

namespace Plow.TransitiveReduction

open Plow Plow.Protocol

variable {tg : TaskGraph}

/-- The invariant the runtime needs: every data edge is ordered by the protocol,
    directly or transitively. Weaker than `WellFormed.edgeCovered`. -/
def CoversHB (p : CounterProtocol tg) : Prop :=
  ∀ e ∈ tg.edges, happensBefore p e.1 e.2

/-- `WellFormed` implies the weaker invariant — this is `protocol_covers_deps`,
    restated so the rest of the file can consume it. -/
theorem coversHB_of_wellFormed (p : CounterProtocol tg) (wf : WellFormed p) :
    CoversHB p :=
  protocol_covers_deps p wf

/-! ## The removal operation -/

/-- `dropWait p t c` removes every occurrence of counter `c` from task `t`'s
    wait list, leaving all other tasks and all other protocol fields alone.

    `List.filter (· ≠ c)` rather than `List.erase` on purpose: `erase` removes
    only the FIRST occurrence, which would model removal (2) incorrectly — the
    emitter's `seen` check removes all repeats. -/
def dropWait (p : CounterProtocol tg) (t : Fin tg.n) (c : CounterId) :
    CounterProtocol tg :=
  { p with waits := fun u => if u = t then (p.waits u).filter (fun x => x ≠ c) else p.waits u }

@[simp] theorem dropWait_resource (p : CounterProtocol tg) (t) (c) (u) :
    (dropWait p t c).resource u = p.resource u := rfl

@[simp] theorem dropWait_streamIdx (p : CounterProtocol tg) (t) (c) (u) :
    (dropWait p t c).streamIdx u = p.streamIdx u := rfl

@[simp] theorem dropWait_succs (p : CounterProtocol tg) (t) (c) (u) :
    (dropWait p t c).succs u = p.succs u := rfl

/-- Resource ordering is untouched: `dropWait` changes only `waits`. -/
theorem resourceOrdered_dropWait (p : CounterProtocol tg) (t) (c) {a b : Fin tg.n} :
    resourceOrdered (dropWait p t c) a b ↔ resourceOrdered p a b := by
  unfold resourceOrdered; simp

/-- A wait that survives the drop. -/
theorem mem_waits_dropWait (p : CounterProtocol tg) (t) (c) {u : Fin tg.n} {d : CounterId}
    (hd : d ∈ p.waits u) (hne : d ≠ c) : d ∈ (dropWait p t c).waits u := by
  dsimp only [dropWait]
  split
  · exact List.mem_filter.mpr ⟨hd, by simpa using hne⟩
  · exact hd

/-- Every counter-gated edge whose witness is some counter OTHER than `c`
    survives the drop. This is the workhorse: it is what keeps the justifying
    path `A→B→C` intact while the edge `A→C` goes away. -/
theorem counterGated_dropWait (p : CounterProtocol tg) (t) (c) {a b : Fin tg.n}
    (h : ∃ d : CounterId, d ∈ p.succs a ∧ d ∈ p.waits b ∧ d ≠ c) :
    counterGated (dropWait p t c) a b := by
  obtain ⟨d, hs, hw, hne⟩ := h
  exact ⟨d, by simpa using hs, mem_waits_dropWait p t c hw hne⟩

/-- Happens-before is preserved for any pair witnessed without using `c`.
    Stated as a hypothesis on the pair rather than proven from `p`, because
    whether a given pair has a `c`-free witness is a property of the concrete
    schedule — exactly what the Rust reduction checks before dropping. -/
theorem happensBefore_dropWait (p : CounterProtocol tg) (t) (c) {a b : Fin tg.n}
    (h : happensBefore p a b)
    (hfree : ∀ x y : Fin tg.n, counterGated p x y →
      (∃ d : CounterId, d ∈ p.succs x ∧ d ∈ p.waits y ∧ d ≠ c) ∨
      resourceOrdered p x y) :
    happensBefore (dropWait p t c) a b := by
  induction h with
  | counter hc =>
      rcases hfree _ _ hc with hfr | hro
      · exact happensBefore.counter (counterGated_dropWait p t c hfr)
      · exact happensBefore.resource ((resourceOrdered_dropWait p t c).mpr hro)
  | resource hr =>
      exact happensBefore.resource ((resourceOrdered_dropWait p t c).mpr hr)
  | trans _ _ ih1 ih2 => exact happensBefore.trans ih1 ih2

/-! ## Removal (1): the transitive reduction -/

/-- **Main soundness theorem for the transitive reduction.**

    Dropping counter `c` from task `t`'s waits preserves `CoversHB` provided:

    * `hpath` — the edge the drop is aimed at is still ordered afterwards, via
      an intermediate task. This is precisely what
      `packet::devbuild::transitive_reduction` establishes: it drops `(a,t)`
      only when another direct successor of `a` reaches `t`.
    * `hfree` — every other ordered pair has a witness that does not rely on the
      dropped wait. In the emitter this holds structurally: `c` is task `a`'s own
      counter (one counter per op, `counter == op index`), and the drop removes it
      from `t`'s list only, so no other pair's witness mentions it.

    Note both hypotheses quantify over the ORIGINAL `p`; the conclusion is about
    the reduced protocol. That direction is the whole point — we are certifying a
    transformation, not assuming its result. -/
theorem tr_preserves_coverage (p : CounterProtocol tg) (t : Fin tg.n) (c : CounterId)
    (hcov : CoversHB p)
    (hfree : ∀ x y : Fin tg.n, counterGated p x y →
      (∃ d : CounterId, d ∈ p.succs x ∧ d ∈ p.waits y ∧ d ≠ c) ∨
      resourceOrdered p x y) :
    CoversHB (dropWait p t c) := by
  intro e he
  exact happensBefore_dropWait p t c (hcov e he) hfree

/-- Iterating the drop: applying the reduction to a whole list of
    `(task, counter)` pairs preserves `CoversHB`, given the per-step side
    condition each time. The Rust pass removes all 69 implied edges in ONE
    sweep; this is the induction that licenses doing so, since for a DAG the
    transitive reduction is unique and reachability-preserving, so each dropped
    edge still has a path in the FINAL graph. -/
theorem tr_preserves_coverage_fold (p : CounterProtocol tg)
    (drops : List (Fin tg.n × CounterId))
    (hcov : CoversHB p)
    (hstep : ∀ (q : CounterProtocol tg) (d : Fin tg.n × CounterId),
      CoversHB q → CoversHB (dropWait q d.1 d.2)) :
    CoversHB (drops.foldl (fun q d => dropWait q d.1 d.2) p) := by
  induction drops generalizing p with
  | nil => simpa using hcov
  | cons d ds ih => exact ih (dropWait p d.1 d.2) (hstep p d hcov)

/-! ## Removal (2): duplicate waits

    Deduplicating a wait list is sound in the STRONG sense — it preserves
    `counterGated`, so `WellFormed.edgeCovered` survives untouched. That is
    because `counterGated` asks only for `d ∈ waits b`, and membership does not
    count occurrences.

    Modelled up to MEMBERSHIP rather than as a concrete `dedup` function, which
    is both weaker as an assumption and a better fit: the emitter drops repeats
    with a running `seen` list, and all that is specified about the result is
    which counters it contains. Any implementation with the same membership is
    covered, including the `List.filter`-based one and `List.eraseDup`. -/

/-- `q` is a wait-deduplication of `p`: same wait MEMBERSHIP everywhere, and
    every other protocol field untouched. -/
structure Dedups (p q : CounterProtocol tg) : Prop where
  waits_iff    : ∀ u d, d ∈ q.waits u ↔ d ∈ p.waits u
  succs_eq     : ∀ u, q.succs u = p.succs u
  resource_eq  : ∀ u, q.resource u = p.resource u
  streamIdx_eq : ∀ u, q.streamIdx u = p.streamIdx u

/-- Dedup preserves counter-gating exactly, in both directions. -/
theorem counterGated_dedup {p q : CounterProtocol tg} (h : Dedups p q) {a b : Fin tg.n} :
    counterGated q a b ↔ counterGated p a b := by
  unfold counterGated
  constructor
  · rintro ⟨d, hs, hw⟩
    exact ⟨d, (h.succs_eq a) ▸ hs, (h.waits_iff b d).mp hw⟩
  · rintro ⟨d, hs, hw⟩
    exact ⟨d, (h.succs_eq a).symm ▸ hs, (h.waits_iff b d).mpr hw⟩

/-- Dedup preserves resource ordering (it touches only `waits`). -/
theorem resourceOrdered_dedup {p q : CounterProtocol tg} (h : Dedups p q) {a b : Fin tg.n} :
    resourceOrdered q a b ↔ resourceOrdered p a b := by
  unfold resourceOrdered
  rw [h.resource_eq a, h.resource_eq b, h.streamIdx_eq a, h.streamIdx_eq b]

/-- **Dedup preserves the STRONG invariant**: every data edge stays *directly*
    counter-gated or resource-ordered. So unlike the transitive reduction, this
    removal needs no weakening of `WellFormed`. -/
theorem dedup_preserves_edgeCovered {p q : CounterProtocol tg} (h : Dedups p q)
    (wf : WellFormed p) :
    ∀ e ∈ tg.edges, counterGated q e.1 e.2 ∨ resourceOrdered q e.1 e.2 := by
  intro e he
  rcases wf.edgeCovered e he with hc | hr
  · exact Or.inl ((counterGated_dedup h).mpr hc)
  · exact Or.inr ((resourceOrdered_dedup h).mpr hr)

/-- …and therefore the weak invariant too. -/
theorem dedup_preserves_coverage {p q : CounterProtocol tg} (h : Dedups p q)
    (wf : WellFormed p) : CoversHB q := by
  intro e he
  rcases dedup_preserves_edgeCovered h wf e he with hc | hr
  · exact happensBefore.counter hc
  · exact happensBefore.resource hr

/-! ## Acyclicity is not disturbed

    Neither removal adds an edge, so the scheduler's topological numbering still
    witnesses forward-pointing base relations, and `happensBefore_acyclic`
    continues to apply. Recorded explicitly because a "reduction" that could
    introduce a cycle would deadlock the runtime rather than merely mis-order it. -/

theorem dropWait_cntForward (p : CounterProtocol tg) (wf : WellFormed p) (t) (c) :
    ∀ a b, counterGated (dropWait p t c) a b → wf.scheduleOrder a < wf.scheduleOrder b := by
  intro a b h
  obtain ⟨d, hs, hw⟩ := h
  refine wf.cntForward a b ⟨d, by simpa using hs, ?_⟩
  dsimp only [dropWait] at hw
  split at hw
  · exact (List.mem_filter.mp hw).1
  · exact hw

theorem dedup_cntForward {p q : CounterProtocol tg} (h : Dedups p q) (wf : WellFormed p) :
    ∀ a b, counterGated q a b → wf.scheduleOrder a < wf.scheduleOrder b := by
  intro a b hc
  exact wf.cntForward a b ((counterGated_dedup h).mp hc)

end Plow.TransitiveReduction
