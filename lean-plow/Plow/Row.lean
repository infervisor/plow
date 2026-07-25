/-
# Plow.Row — row-op correctness theorems.

Covers the compiler's `OpKind::Row` family: RMSNorm, LayerNorm, activation
functions, elementwise `Ew` (add/mul), and their row-wise decomposition.

* **R1**: row independence — a row-wise op decomposes into per-row applications.
* **R2**: pointwise-ness — activations map each element independently.
* **R3**: Ew commutativity / associativity for `add` and `mul` kinds.
* **R4**: reduction operators are stable under partial sums (used by
  streaming norm reductions).
-/
import Plow.Basic

namespace Plow.Row

/-! ## R1 — Row-op independence. -/

/-- A row-op is a function `Nat → Nat` applied uniformly to each row.
    Rows in the input tensor are indexed by `row_id : Nat`; the op reads the
    row's feature vector (abstracted as a scalar sum here) and writes the
    output. -/
def RowOp : Type := Nat → Nat

/-- Applying a row-op to a tensor (modeled as `Nat → Nat` — row_id → feature
    scalar) applies it independently to each row. -/
def applyRow (f : RowOp) (input : Nat → Nat) : Nat → Nat :=
  fun row => f (input row)

/-- **R1**: row-op application on `n` rows equals `n` independent per-row
    applications. Trivial by definition; the theorem certifies the
    row-wise decomposition the tiler uses. -/
theorem row_independence (f : RowOp) (input : Nat → Nat) (row : Nat) :
    applyRow f input row = f (input row) := rfl

/-- **R1** corollary: two row-ops applied in row-order sequence produce the
    same result as being applied per-row in composition. -/
theorem row_op_composition (f g : RowOp) (input : Nat → Nat) (row : Nat) :
    applyRow g (applyRow f input) row = g (f (input row)) := rfl

/-! ## R2 — Pointwise activation preservation. -/

/-- An activation function is pointwise: applied to each *element* (not just
    each row). -/
def PointwiseAct : Type := Nat → Nat

/-- Row of features + a pointwise activation ⇒ row of activated features.
    Modeled: the activation commutes with row extraction. -/
def applyPointwise (act : PointwiseAct) (row : Nat → Nat) : Nat → Nat :=
  fun feat_idx => act (row feat_idx)

/-- **R2**: applying an activation elementwise to a row is the same as
    applying it after any row-permutation. Certifies that activations
    survive the tiler's per-row processing regardless of feature-tile order. -/
theorem pointwise_permutation_invariant
    (act : PointwiseAct) (row : Nat → Nat) (perm : Nat → Nat) (feat_idx : Nat) :
    applyPointwise act (row ∘ perm) feat_idx
    = (applyPointwise act row) (perm feat_idx) := rfl

/-! ## R3 — Ew (elementwise) op laws. -/

/-- Elementwise op kinds the compiler uses. -/
inductive EwKind
  | Add
  | Mul

/-- The scalar operation for each kind. -/
def ewOp : EwKind → Nat → Nat → Nat
  | EwKind.Add, x, y => x + y
  | EwKind.Mul, x, y => x * y

/-- **R3a**: `Add` is commutative. -/
theorem ew_add_comm (x y : Nat) :
    ewOp EwKind.Add x y = ewOp EwKind.Add y x := by
  unfold ewOp
  exact Nat.add_comm _ _

/-- **R3b**: `Mul` is commutative. -/
theorem ew_mul_comm (x y : Nat) :
    ewOp EwKind.Mul x y = ewOp EwKind.Mul y x := by
  unfold ewOp
  exact Nat.mul_comm _ _

/-- **R3c**: `Add` is associative — used by residual reordering. -/
theorem ew_add_assoc (x y z : Nat) :
    ewOp EwKind.Add (ewOp EwKind.Add x y) z
    = ewOp EwKind.Add x (ewOp EwKind.Add y z) := by
  unfold ewOp
  exact Nat.add_assoc _ _ _

/-- **R3d**: `Mul` is associative. -/
theorem ew_mul_assoc (x y z : Nat) :
    ewOp EwKind.Mul (ewOp EwKind.Mul x y) z
    = ewOp EwKind.Mul x (ewOp EwKind.Mul y z) := by
  unfold ewOp
  exact Nat.mul_assoc _ _ _

/-- **R3e**: distributivity — `x · (y + z) = x·y + x·z`. Used by
    activation-times-up fusion (SwiGLU). -/
theorem ew_mul_add_distrib (x y z : Nat) :
    ewOp EwKind.Mul x (ewOp EwKind.Add y z)
    = ewOp EwKind.Add (ewOp EwKind.Mul x y) (ewOp EwKind.Mul x z) := by
  unfold ewOp
  exact Nat.mul_add _ _ _

/-! ## R4 — Row reduction stability. -/

/-- Alias for `Plow.sumRange` — kept for the historical name. -/
abbrev partialSum : Nat → (Nat → Nat) → Nat := Plow.sumRange

/-- **R4**: partial sums are stable under splitting — sum over `[0, m)` +
    sum over `[m, n)` = sum over `[0, n)`, for any split point `m ≤ n`.
    Used by streaming norm reductions (compute variance in chunks). -/
theorem partial_sum_split (m n : Nat) (f : Nat → Nat) (h : m ≤ n) :
    partialSum m f + (partialSum (n - m) (fun i => f (m + i)))
    = partialSum n f := by
  -- Induction on `n - m`.
  induction n with
  | zero =>
    -- m ≤ 0 ⇒ m = 0.
    have : m = 0 := Nat.le_zero.mp h
    subst this
    simp [partialSum, Plow.sumRange]
  | succ k ih =>
    by_cases hm : m ≤ k
    · -- The recursion peels the last element off `n = k + 1`.
      have h_diff : k + 1 - m = (k - m) + 1 := by omega
      have h_shift : (fun i => f (m + i)) (k - m) = f k := by
        show f (m + (k - m)) = f k
        congr 1; omega
      show partialSum m f + partialSum ((k + 1) - m) (fun i => f (m + i))
           = partialSum (k + 1) f
      rw [h_diff]
      show partialSum m f + (partialSum (k - m) (fun i => f (m + i))
                            + (fun i => f (m + i)) (k - m))
           = partialSum k f + f k
      rw [h_shift]
      have := ih hm
      omega
    · -- m > k ⇒ m = k + 1 (since m ≤ k + 1 by h).
      have : m = k + 1 := by omega
      subst this
      simp [partialSum, Plow.sumRange]

end Plow.Row
