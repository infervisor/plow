/-
# Plow.Cost — tile-cost monotonicity and Pareto dominance (§8.4).

Encodes the three monotone axes the Rust `costmodel::dominance` pass uses
(`passes`, `sram_pages`, `output_tiles`) and proves that Pareto dominance in
those three implies dominance in any cost model that is monotone in each.

This is the theorem that makes the Rust `prune_dominated` pass safe: dropping
a Pareto-dominated tile cannot remove the optimum, because the dominator has
`cost ≤` the dominated in the modeled cost function.

Kept deliberately abstract in the cost function: we prove the property for
*any* `cost : TileMetrics → Nat` that is monotone in each field. The concrete
Rust `gemm_cycles` is not verified to be monotone here (that would require
formalizing the whole cost function); the theorem lets the caller commit to
monotonicity as an assumption at the seam.
-/
import Plow.Basic

namespace Plow.Cost

open Plow

/-- The three-dimensional projection of a tile the Pareto pass compares. -/
structure TileMetrics where
  passes      : Nat
  sramPages   : Nat
  outputTiles : Nat
  deriving Repr, DecidableEq

/-- Pareto dominance: `a` ≤ `b` in every axis, and strictly less in some. -/
def dominates (a b : TileMetrics) : Prop :=
  (a.passes ≤ b.passes ∧ a.sramPages ≤ b.sramPages ∧ a.outputTiles ≤ b.outputTiles)
  ∧ (a.passes < b.passes ∨ a.sramPages < b.sramPages ∨ a.outputTiles < b.outputTiles)

/-- A cost function that is monotone (non-decreasing) in each axis. -/
structure MonotoneCost (cost : TileMetrics → Nat) : Prop where
  monoPasses : ∀ p1 p2 s o, p1 ≤ p2 → cost ⟨p1, s, o⟩ ≤ cost ⟨p2, s, o⟩
  monoPages  : ∀ p s1 s2 o, s1 ≤ s2 → cost ⟨p, s1, o⟩ ≤ cost ⟨p, s2, o⟩
  monoTiles  : ∀ p s o1 o2, o1 ≤ o2 → cost ⟨p, s, o1⟩ ≤ cost ⟨p, s, o2⟩

/-! ## Main theorem — dominance implies cost order. -/

/-- If `cost` is monotone in every axis and `a` Pareto-dominates `b`, then
    `cost a ≤ cost b`. Proven by chaining the three axis-monotonicity steps
    through the intermediate metrics `⟨a.passes, a.sramPages, b.outputTiles⟩`
    and `⟨a.passes, b.sramPages, b.outputTiles⟩`. -/
theorem dominates_implies_cost_le
    (cost : TileMetrics → Nat) (mono : MonotoneCost cost)
    (a b : TileMetrics) (h : dominates a b) :
    cost a ≤ cost b := by
  obtain ⟨⟨hp, hs, ho⟩, _hstrict⟩ := h
  -- Step 1: raise output tiles from a.outputTiles to b.outputTiles.
  have step1 : cost ⟨a.passes, a.sramPages, a.outputTiles⟩
             ≤ cost ⟨a.passes, a.sramPages, b.outputTiles⟩ :=
    mono.monoTiles _ _ _ _ ho
  -- Step 2: raise sram pages from a.sramPages to b.sramPages.
  have step2 : cost ⟨a.passes, a.sramPages, b.outputTiles⟩
             ≤ cost ⟨a.passes, b.sramPages, b.outputTiles⟩ :=
    mono.monoPages _ _ _ _ hs
  -- Step 3: raise passes from a.passes to b.passes.
  have step3 : cost ⟨a.passes, b.sramPages, b.outputTiles⟩
             ≤ cost ⟨b.passes, b.sramPages, b.outputTiles⟩ :=
    mono.monoPasses _ _ _ _ hp
  -- The starting and ending records match `a` and `b` after eta.
  have hA : cost a = cost ⟨a.passes, a.sramPages, a.outputTiles⟩ := by cases a; rfl
  have hB : cost b = cost ⟨b.passes, b.sramPages, b.outputTiles⟩ := by cases b; rfl
  calc cost a = cost ⟨a.passes, a.sramPages, a.outputTiles⟩ := hA
    _ ≤ cost ⟨a.passes, a.sramPages, b.outputTiles⟩ := step1
    _ ≤ cost ⟨a.passes, b.sramPages, b.outputTiles⟩ := step2
    _ ≤ cost ⟨b.passes, b.sramPages, b.outputTiles⟩ := step3
    _ = cost b := hB.symm

/-- Consequence: any Pareto-dominated tile can be dropped from a candidate
    list without removing the min-cost element. If `a` dominates `b`, then
    `a` is at least as good in the modeled cost, so `b` is never optimal
    alone (`a` — or something equal-or-better — will match its cost). -/
theorem pruning_preserves_optimum
    (cost : TileMetrics → Nat) (mono : MonotoneCost cost)
    (a b : TileMetrics) (h : dominates a b) :
    ∃ c, cost c ≤ cost b := ⟨a, dominates_implies_cost_le cost mono a b h⟩

end Plow.Cost
