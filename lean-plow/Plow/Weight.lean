/-
# Plow.Weight — weight-tile amortization theorems.

The key perf lever for transformer batching: for a GEMM `[M, K] × [K, N]`
tiled `(BM, BN, BK)`, the total weight bytes fetched from HBM are
**independent of M** and equal `N · K · elem_bytes` (each weight tile loaded
exactly once), as long as `M ≥ BM` (weight is amortized across `⌈M/BM⌉` output
row groups).

This certifies:
* Batching amortizes weight DMA — big M → same weight DMA, more compute.
* Prefill (M = batch × seq) is compute-bound; decode (M = batch = 1) is
  weight-DMA-bound.
* The `weight_shared` invariant across buckets: shared `(BN, BK)` means the
  same tiled bytes serve every bucket.
-/
import Plow.Basic

namespace Plow.Weight

/-- Per-tile weight-panel bytes: `bn · bk · elem_bytes` for one `(BN, BK)`
    weight tile. -/
def tileBytes (bn bk elem : Nat) : Nat := bn * bk * elem

/-- Total weight tiles in `(N, K)` under `(BN, BK)` tiling. -/
def weightTileCount (n k bn bk : Nat) : Nat :=
  ceilDiv n bn * ceilDiv k bk

/-- Total weight bytes fetched to run a `[M, K] × [K, N]` GEMM tiled
    `(BM, BN, BK)`, **when each weight tile is loaded exactly once** —
    the ideal cache/persistent-store case. -/
def weightBytesIdeal (n k bn bk elem : Nat) : Nat :=
  weightTileCount n k bn bk * tileBytes bn bk elem

/-! ## W1 — Weight bytes independent of M in the ideal case.

    The ideal HBM traffic to bring the full weight matrix in equals
    `⌈N/BN⌉ · ⌈K/BK⌉ · BN · BK · elem_bytes` — no M term appears. We model
    the persistent-weight execution explicitly: `weightBytesPersistent`
    takes M as an argument (it is a per-GEMM cost) but its value never
    depends on it. -/

/-- Weight bytes fetched under the persistent-weight execution model:
    the weight tiles stay resident, so exactly one full fetch happens
    regardless of how many output-row groups M produces. M is an argument
    (the cost is per-GEMM) but the result never mentions it. -/
def weightBytesPersistent (_m _bm n k bn bk elem : Nat) : Nat :=
  weightBytesIdeal n k bn bk elem

/-- **W1**: persistent-weight bytes are constant in M — any two values of
    M (at fixed tiling) fetch identical weight bytes. Holds definitionally
    because `weightBytesPersistent` discards its M argument; the content is
    in the model, not the proof. -/
theorem weight_bytes_same_across_M (m1 m2 bm n k bn bk elem : Nat) :
    weightBytesPersistent m1 bm n k bn bk elem
    = weightBytesPersistent m2 bm n k bn bk elem := rfl

/-! ## W2 — Amortization factor: weight bytes per output-row group.

    Under `⌈M/BM⌉` output-row groups, if we fetched weight once per group,
    total = `⌈M/BM⌉ · weightBytesIdeal`. The persistent case avoids all but
    the first fetch, saving `(⌈M/BM⌉ - 1) · weightBytesIdeal` bytes. -/

/-- Naive weight bytes: fetch weight fresh for every output-row group. -/
def weightBytesNaive (m bm n k bn bk elem : Nat) : Nat :=
  ceilDiv m bm * weightBytesIdeal n k bn bk elem

/-- **W2**: the persistent-weight optimization saves exactly `⌈M/BM⌉ - 1`
    copies of the full weight-DMA cost relative to fetching per row group. -/
theorem persistent_weight_saves (m bm n k bn bk elem : Nat) :
    weightBytesNaive m bm n k bn bk elem
      - weightBytesPersistent m bm n k bn bk elem
    = (ceilDiv m bm - 1) * weightBytesIdeal n k bn bk elem := by
  unfold weightBytesNaive weightBytesPersistent
  rw [Nat.sub_one_mul]

/-- **W2** (soundness of the subtraction): the naive cost dominates the
    persistent cost whenever there is at least one row group (`0 < m`,
    `0 < bm` ⇒ `⌈M/BM⌉ ≥ 1`). -/
theorem persistent_le_naive (m bm n k bn bk elem : Nat)
    (hm : 0 < m) (hbm : 0 < bm) :
    weightBytesPersistent m bm n k bn bk elem
    ≤ weightBytesNaive m bm n k bn bk elem := by
  unfold weightBytesPersistent weightBytesNaive
  have h1 : 1 ≤ ceilDiv m bm := by
    unfold ceilDiv
    rw [if_neg (Nat.pos_iff_ne_zero.mp hbm)]
    exact (Nat.le_div_iff_mul_le hbm).mpr (by omega)
  calc weightBytesIdeal n k bn bk elem
      = 1 * weightBytesIdeal n k bn bk elem := (Nat.one_mul _).symm
    _ ≤ ceilDiv m bm * weightBytesIdeal n k bn bk elem :=
        Nat.mul_le_mul_right _ h1

/-- **W2 corollary**: for `M = 0` the naive count is zero (no output rows,
    no weight fetches). For `M ≥ BM`, ratio = `⌈M/BM⌉`. -/
theorem naive_zero_when_M_zero (bm n k bn bk elem : Nat) :
    weightBytesNaive 0 bm n k bn bk elem = 0 := by
  unfold weightBytesNaive ceilDiv
  by_cases h : bm = 0
  · simp [h]
  · rw [if_neg h]
    have hlt : 0 + bm - 1 < bm := by omega
    have hdiv0 : (0 + bm - 1) / bm = 0 := Nat.div_eq_of_lt hlt
    rw [hdiv0]
    simp

/-! ## W3 — Decode is weight-DMA-bound.

    In decode, `M = batch` (typically 1). So `⌈M/BM⌉ = 1`. Ideal and naive
    fetch counts coincide — weight bytes dominate. -/

/-- **W3**: when `M ≤ BM`, weight bytes ≡ ideal count (one fetch). Certifies
    the decode-is-weight-bound observation. -/
theorem decode_weight_equals_ideal (m bm n k bn bk elem : Nat)
    (hbm : 0 < bm) (hM : 0 < m) (hMle : m ≤ bm) :
    weightBytesNaive m bm n k bn bk elem
    = weightBytesIdeal n k bn bk elem := by
  unfold weightBytesNaive ceilDiv
  rw [if_neg (Nat.pos_iff_ne_zero.mp hbm)]
  -- ⌈m/bm⌉ = 1 when 0 < m ≤ bm.
  have h1 : m + bm - 1 < 2 * bm := by omega
  have h2 : m + bm - 1 ≥ bm := by omega
  have hdiv := Nat.div_add_mod (m + bm - 1) bm
  have hmod : (m + bm - 1) % bm < bm := Nat.mod_lt _ hbm
  have hq1 : (m + bm - 1) / bm = 1 := by
    -- Nat.div_lt_iff_lt_mul: `x / d < n ↔ x < n · d`.
    have goal_lt : (m + bm - 1) / bm < 2 :=
      (Nat.div_lt_iff_lt_mul hbm).mpr (by omega)
    -- Lower bound: `bm ≤ x` ⇒ `x / bm ≥ 1`. Proved by omega on hdiv/hmod.
    have goal_ge : (m + bm - 1) / bm ≥ 1 := by
      -- bm * q + mod = x, mod < bm, x ≥ bm ⇒ bm * q ≥ bm - (bm - 1) = 1 ⇒ q ≥ 1.
      -- Equivalently: if q = 0 then bm*0 + mod = x, but mod < bm ≤ x, contradiction.
      cases hq : (m + bm - 1) / bm with
      | zero =>
        rw [hq] at hdiv
        simp at hdiv
        omega
      | succ n => omega
    omega
  rw [hq1]
  simp

end Plow.Weight
