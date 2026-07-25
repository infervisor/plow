/-
# Plow.Basic — shared abstractions for the Lean side of plow.

Mirrors enough of `crates/schedule` and `crates/schedule/src/memory.rs` to
state the universal lemmas in §2 and §6.2 of the formal-verification plan.
Kept deliberately minimal: only the structure the proofs use.
-/

namespace Plow

/-- Identifier of a task in the per-bucket TaskGraph. -/
abbrev TaskId := Nat

/-- Identifier of a synchronization counter. -/
abbrev CounterId := Nat

/-- Identifier of an execution resource (an SM, a DMA engine, etc). -/
abbrev ResourceId := Nat

/-- Simulator cycle (the `Cycle` alias used by `passes::Schedule`). -/
abbrev Cycle := Nat

/-- A buffer's lifetime/placement class (mirrors `BufClass`). -/
inductive BufClass
  | Persistent
  | RequestIo
  | Scratch
  | Growable
  deriving DecidableEq, Repr

/-- Half-open cycle interval `[start, stop)`. -/
structure LiveInterval where
  start : Cycle
  stop  : Cycle
  deriving Repr

/-- Two intervals overlap iff each starts before the other ends. -/
def LiveInterval.overlap (a b : LiveInterval) : Prop :=
  a.start < b.stop ∧ b.start < a.stop

/-- Byte range `[off, off + size)` overlaps another byte range. -/
def bytesOverlap (off1 sz1 off2 sz2 : Nat) : Prop :=
  off1 < off2 + sz2 ∧ off2 < off1 + sz1

/-! ## Shared arithmetic helpers.

    Consolidated here so every downstream module (`TilePartition`, `Attn`,
    `KvPerf`, etc.) uses the same definitions instead of re-deriving them. -/

/-- Ceiling division `⌈a / b⌉`. Zero when `b = 0`. -/
def ceilDiv (a b : Nat) : Nat :=
  if b = 0 then 0 else (a + b - 1) / b

/-- Core inequality: `b · ⌈a/b⌉ ≥ a` when `b > 0`. -/
theorem le_ceilDiv_mul (a b : Nat) (hb : 0 < b) :
    a ≤ b * ceilDiv a b := by
  unfold ceilDiv
  rw [if_neg (Nat.pos_iff_ne_zero.mp hb)]
  have hdiv : b * ((a + b - 1) / b) + (a + b - 1) % b = a + b - 1 :=
    Nat.div_add_mod (a + b - 1) b
  have hmod : (a + b - 1) % b < b := Nat.mod_lt _ hb
  omega

/-- Swapped form: `a ≤ ⌈a/b⌉ · b`. -/
theorem le_ceilDiv_mul' (a b : Nat) (hb : 0 < b) :
    a ≤ ceilDiv a b * b := by
  have := le_ceilDiv_mul a b hb
  rw [Nat.mul_comm] at this
  exact this

/-! ## Byte-view abstraction — byte-indexed value function. -/

/-- A tensor as a byte-indexed function `Nat → Nat`. Two views are equal
    when they agree at every offset. Used across `AliasPerf`, `CrossOpPerf`,
    etc. -/
def ByteView : Type := Nat → Nat

def viewEq (v w : ByteView) : Prop := ∀ i, v i = w i

@[refl] theorem viewEq_refl (v : ByteView) : viewEq v v := fun _ => rfl

theorem viewEq_symm {v w : ByteView} (h : viewEq v w) : viewEq w v :=
  fun i => (h i).symm

theorem viewEq_trans {v w u : ByteView}
    (h1 : viewEq v w) (h2 : viewEq w u) : viewEq v u :=
  fun i => (h1 i).trans (h2 i)

/-! ## Fold over `[0, n)` — abstract sum. -/

/-- Sum `f` over `[0, n)`. Used by `SplitK` (partial sums) and `Row`
    (streaming reductions). -/
def sumRange : Nat → (Nat → Nat) → Nat
  | 0, _ => 0
  | n + 1, f => sumRange n f + f n

end Plow
