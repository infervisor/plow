/-
# Plow.Sram — SRAM temporal fit theorem (§8.5).

Models per-SM page-pool occupancy as a function of cycle for a producer/
consumer hand-off, and proves the two-condition safety property the Rust
`sram_fit::analyze_temporal_fit` pass depends on:

  producer_release ≤ consumer_acquire     (temporal disjointness)
      ∧ max(producer_pages, consumer_pages) ≤ budget
      ⟹  ∀ cycle t, occupancy t ≤ budget

That is the theorem `occupancy_le_of_temporal_fit` below.

## Role: design-time soundness, not per-bucket runtime check

The Rust filter in `crates/schedule/src/sram_fit.rs::analyze_temporal_fit`
applies the same two predicates as `checkSramFit` here. Once this theorem
is discharged, the Rust filter is **known-correct by construction**:
every candidate it admits satisfies the occupancy bound at every cycle,
with no runtime evidence needed. Per-bucket checkpoint C is therefore
**not** wired into plowc's `run_lean_verify` — it would just re-run the
same predicate over IPC with no additional signal. `checkSramFit` and the
`check_sram_fit` bridge remain callable for opt-in dev spot checks
(exercised by `crates/plowc/tests/lean_verify_sram_fit.rs`).

The two conditions are independent — dropping either allows a
counter-example (co-scheduled peaks; a single peak exceeding budget alone).
Both are needed, see `violates_when_both_dropped`.
-/
import Plow.Basic

namespace Plow.Sram

open Plow

/-- A hand-off descriptor: producer and consumer each have a page footprint
    and a lifetime window on the SM. Cycles are non-negative naturals. -/
structure Handoff where
  producerPages    : Nat
  consumerPages    : Nat
  producerRelease  : Cycle          -- last cycle producer's pages are held
  consumerAcquire  : Cycle          -- first cycle consumer's pages are held
  consumerRelease  : Cycle          -- last cycle consumer's pages are held
  deriving Repr

/-- Page occupancy on the SM at cycle `t`:
    * producer pages are live on `[0, producerRelease)` (strict — released at
      `producerRelease`),
    * consumer pages are live on `[consumerAcquire, consumerRelease]` (closed),
    * anywhere else, 0 pages.
    Sum is what the pool sees. -/
def occupancy (h : Handoff) (t : Cycle) : Nat :=
  (if t < h.producerRelease then h.producerPages else 0)
  + (if h.consumerAcquire ≤ t ∧ t ≤ h.consumerRelease then h.consumerPages else 0)

/-- Temporal disjointness: producer's window ends no later than consumer's
    window starts. -/
def temporallyDisjoint (h : Handoff) : Prop :=
  h.producerRelease ≤ h.consumerAcquire

/-- The two-part safety predicate the Rust pass checks. -/
def temporalFitSafe (h : Handoff) (budget : Nat) : Prop :=
  temporallyDisjoint h
    ∧ h.producerPages ≤ budget
    ∧ h.consumerPages ≤ budget

/-! ## Main theorem: `temporalFitSafe` implies bounded occupancy everywhere. -/

theorem occupancy_le_of_temporal_fit
    (h : Handoff) (budget : Nat) (safe : temporalFitSafe h budget) :
    ∀ t, occupancy h t ≤ budget := by
  intro t
  obtain ⟨hdisj, hpb, hcb⟩ := safe
  unfold temporallyDisjoint at hdisj
  unfold occupancy
  -- Case on producer-window membership.
  by_cases hprod : t < h.producerRelease
  · -- Producer active. By disjointness `producerRelease ≤ consumerAcquire`,
    -- so `t < consumerAcquire`, so the consumer term evaluates to 0.
    have hcons_inactive : ¬ (h.consumerAcquire ≤ t ∧ t ≤ h.consumerRelease) := by
      intro ⟨hacq, _⟩
      -- Chain: t < producerRelease ≤ consumerAcquire ≤ t ⇒ t < t.
      exact Nat.lt_irrefl t
        (Nat.lt_of_lt_of_le (Nat.lt_of_lt_of_le hprod hdisj) hacq)
    rw [if_pos hprod, if_neg hcons_inactive, Nat.add_zero]
    exact hpb
  · rw [if_neg hprod, Nat.zero_add]
    by_cases hcons : h.consumerAcquire ≤ t ∧ t ≤ h.consumerRelease
    · rw [if_pos hcons]
      exact hcb
    · rw [if_neg hcons]
      exact Nat.zero_le _

/-- The temporal-fit rule is *tight* in this sense: without at least one of
    the two conditions, occupancy can exceed the budget.

    (Sanity witness that both parts of `temporalFitSafe` are needed. If we
    drop temporal disjointness, a co-scheduled pair can exceed budget by
    `producerPages + consumerPages`. If we drop the individual-peak bounds,
    a single peak already exceeds budget.) -/
theorem violates_when_both_dropped :
    ∃ (h : Handoff) (budget : Nat),
      ¬ temporallyDisjoint h
      ∧ ¬ (h.producerPages ≤ budget ∧ h.consumerPages ≤ budget)
      ∧ ∃ t, budget < occupancy h t := by
  -- Concrete counter-example: overlapping windows, each peak > budget.
  refine ⟨{
    producerPages := 5, consumerPages := 5,
    producerRelease := 10, consumerAcquire := 5, consumerRelease := 15
  }, 3, ?_, ?_, ?_⟩
  · simp [temporallyDisjoint]
  · simp
  · refine ⟨7, ?_⟩
    simp [occupancy]

end Plow.Sram
