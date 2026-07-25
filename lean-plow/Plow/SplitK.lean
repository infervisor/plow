/-
# Plow.SplitK — split-K correctness + occupancy monotonicity (§A2 A3).

* **A2**: reducing a sum by any partition (split-K across SMs) equals the
  same sum evaluated sequentially — associativity/commutativity of `+`.
* **A3**: SRAM occupancy is monotone in the per-tile page footprint.
-/
import Plow.Basic

namespace Plow.SplitK

/-! ## A2 — Split-K reduction correctness.

    Uses `Plow.sumRange` from `Plow.Basic` for the underlying summation. -/

/-- Splitting the summation at index `n`: `Σ_{i<n+m} f(i) = Σ_{i<n} f(i)
    + Σ_{i<m} f(n+i)`. Proven by induction on `m`. -/
theorem sumRange_split (n m : Nat) (f : Nat → Nat) :
    sumRange (n + m) f = sumRange n f + sumRange m (fun i => f (n + i)) := by
  induction m with
  | zero =>
    show sumRange n f = sumRange n f + sumRange 0 (fun i => f (n + i))
    simp [sumRange]
  | succ k ih =>
    show sumRange (n + (k + 1)) f
       = sumRange n f + sumRange (k + 1) (fun i => f (n + i))
    -- LHS = sumRange (n+k+1) f = sumRange (n+k) f + f (n+k).
    -- RHS = sumRange n f + (sumRange k (fun i => f (n+i)) + f (n+k)).
    -- IH gives the first pair; algebra closes.
    have h1 : sumRange (n + (k + 1)) f = sumRange (n + k) f + f (n + k) := by
      show sumRange ((n + k) + 1) f = sumRange (n + k) f + f (n + k)
      rfl
    have h2 : sumRange (k + 1) (fun i => f (n + i))
              = sumRange k (fun i => f (n + i)) + f (n + k) := by
      show sumRange k (fun i => f (n + i)) + (fun i => f (n + i)) k
           = sumRange k (fun i => f (n + i)) + f (n + k)
      rfl
    rw [h1, ih, h2]
    omega

/-- **A2** (2-way split): summing `[0, N)` is the same as summing two halves
    when the split point is in range. -/
theorem split_k_two_way (N split : Nat) (f : Nat → Nat)
    (hsplit : split ≤ N) :
    sumRange N f
    = sumRange split f + sumRange (N - split) (fun i => f (split + i)) := by
  have h : split + (N - split) = N := by omega
  calc sumRange N f
      = sumRange (split + (N - split)) f := by rw [h]
    _ = sumRange split f + sumRange (N - split) (fun i => f (split + i)) :=
        sumRange_split split (N - split) f

/-! ## A3 — Occupancy monotonicity. -/

/-- SM's page pool occupancy at a moment: `n` simultaneous live tiles, each
    with `p` pages. -/
def occupancy (n p : Nat) : Nat := n * p

/-- **A3**: occupancy is monotone in per-tile page footprint. Shrinking a
    tile's `p` keeps occupancy ≤ the original. -/
theorem occupancy_mono_pages (n p p' : Nat) (h : p' ≤ p) :
    occupancy n p' ≤ occupancy n p := by
  unfold occupancy
  exact Nat.mul_le_mul_left n h

/-- Occupancy is monotone in the number of live tiles. -/
theorem occupancy_mono_count (n n' p : Nat) (h : n' ≤ n) :
    occupancy n' p ≤ occupancy n p := by
  unfold occupancy
  exact Nat.mul_le_mul_right p h

/-- If a tile fits under budget, every strictly-smaller-footprint tile also
    fits. Used by the tile extractor to certify that shrinking is always
    safe. -/
theorem fits_of_smaller (n p p' budget : Nat)
    (h : occupancy n p ≤ budget) (hle : p' ≤ p) :
    occupancy n p' ≤ budget :=
  Nat.le_trans (occupancy_mono_pages n p p' hle) h

end Plow.SplitK
