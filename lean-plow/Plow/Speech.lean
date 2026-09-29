/-
# Plow.Speech — facts the speech FP32 packets (encoder.pkt, codec.pkt, s3gen.pkt) rely on.

* Residue bands: `plow_asset::logical_effects` gives a strided `CopyColsF32` / `GatherRowsF32`
  access the byte interval `[4 lo, 4 hi)` of its stride-residue band instead of the whole
  tensor. `checkConflicts` only orders byte-overlapping accesses, so that is sound only if
  accesses whose bands do not overlap touch no common element — `band_disjoint`.
* `DenseGemmF32`'s LayerNorm prologue (flag bit 5): RowStatsF32 + prologue computes, per A
  element, exactly the value LayerNormF32 would have stored, in the same operation order —
  `ln_prologue_eq`, over arbitrary (non-associative) arithmetic.
* `Conv1dF32` `row_scale`: `residual + row_scale * post(conv)` equals the unfused
  BinaryF32 mul (row vector) then BinaryF32 add, assuming only IEEE commutativity of `*`
  and `+` — `conv_row_scale_residual_eq`.
-/
namespace Plow.Speech

/-! ## Residue bands -/

/-- Element index touched by a banded access: item `b` at item stride `stride * q`
    (the Rust side requires `item_stride % stride = 0`), row `r`, column `lo + c`. -/
def bandElem (stride q lo b r c : Nat) : Nat := b * (stride * q) + r * stride + (lo + c)

theorem band_elem_residue (stride q lo hi b r c : Nat) (hhi : hi ≤ stride) (hc : lo + c < hi) :
    bandElem stride q lo b r c % stride = lo + c := by
  have hlt : lo + c < stride := Nat.lt_of_lt_of_le hc hhi
  have e : bandElem stride q lo b r c = stride * (b * q + r) + (lo + c) := by
    simp only [bandElem, Nat.mul_add, Nat.mul_comm, Nat.mul_left_comm]
  rw [e, Nat.mul_add_mod, Nat.mod_eq_of_lt hlt]

/-- Two accesses under one stride whose residue bands are disjoint share no element. -/
theorem band_disjoint (stride q₁ q₂ lo₁ hi₁ lo₂ hi₂ b₁ r₁ c₁ b₂ r₂ c₂ : Nat)
    (h₁ : hi₁ ≤ stride) (h₂ : hi₂ ≤ stride)
    (hc₁ : lo₁ + c₁ < hi₁) (hc₂ : lo₂ + c₂ < hi₂)
    (hdis : hi₁ ≤ lo₂ ∨ hi₂ ≤ lo₁) :
    bandElem stride q₁ lo₁ b₁ r₁ c₁ ≠ bandElem stride q₂ lo₂ b₂ r₂ c₂ := by
  intro heq
  have e₁ := band_elem_residue stride q₁ lo₁ hi₁ b₁ r₁ c₁ h₁ hc₁
  have e₂ := band_elem_residue stride q₂ lo₂ hi₂ b₂ r₂ c₂ h₂ hc₂
  rw [heq] at e₁
  omega

/-- Non-overlapping byte intervals `[4 lo, 4 hi)` (what `checkConflicts` compares) mean
    disjoint bands. -/
theorem band_bytes_disjoint (lo₁ hi₁ lo₂ hi₂ : Nat)
    (h : ¬ (4 * lo₁ < 4 * lo₂ + 4 * (hi₂ - lo₂) ∧ 4 * lo₂ < 4 * lo₁ + 4 * (hi₁ - lo₁)))
    (w₁ : lo₁ < hi₁) (w₂ : lo₂ < hi₂) : hi₁ ≤ lo₂ ∨ hi₂ ≤ lo₁ := by
  omega

/-! ## Opaque FP32 arithmetic -/

/-- Rounded FP32 operations with no algebraic laws: equalities below hold for IEEE
    arithmetic because both sides perform the same operations on the same operands. -/
structure Arith (α : Type) where
  add : α → α → α
  sub : α → α → α
  mul : α → α → α
  bf16 : α → α
  one : α
  zero : α

/-! ## LayerNorm prologue -/

/-- LayerNormF32's per-row statistics (mean, 1/sqrt(var+eps)); RowStatsF32 runs the same
    code (`d_layernorm_f32` with `stats`), so both are this one function of the row. -/
def lnValue {α : Type} (A : Arith α) (round : Bool) (x mean inv : α)
    (gamma beta : Option α) : α :=
  let v := A.mul (A.sub x mean) inv
  let v := A.add (A.mul v (gamma.getD A.one)) (beta.getD A.zero)
  if round then A.bf16 v else v

/-- LayerNormF32 output element `(r, k)` for row statistics `stats`. -/
def layerNorm {α : Type} (A : Arith α) (stats : Nat → α × α) (round : Bool)
    (x : Nat → Nat → α) (gamma beta : Nat → Option α) (r k : Nat) : α :=
  lnValue A round (x r k) (stats r).1 (stats r).2 (gamma k) (beta k)

/-- The GEMM's A operand under the prologue: row `r` of the M-row block reads source row
    `a_row0 + r` and statistics row `a_row0 + r` (`SpLn::apply`). -/
def prologueA {α : Type} (A : Arith α) (statsT : Nat → α × α) (roundBit6 : Bool)
    (x : Nat → Nat → α) (gamma beta : Nat → Option α) (aRow0 r k : Nat) : α :=
  lnValue A roundBit6 (x (aRow0 + r) k) (statsT (aRow0 + r)).1 (statsT (aRow0 + r)).2
    (gamma k) (beta k)

/-- The binding the fusion requires: the stats tensor is RowStatsF32 over the same source
    rows with LayerNormF32's flags/eps (so `statsT` agrees with LayerNorm's statistics on every
    row the GEMM reads), and bit 6 equals LayerNormF32's bf16 flag. -/
theorem ln_prologue_eq {α : Type} (A : Arith α) (stats statsT : Nat → α × α)
    (lnRound roundBit6 : Bool) (x : Nat → Nat → α) (gamma beta : Nat → Option α)
    (aRow0 M : Nat) (hstats : ∀ r, r < M → statsT (aRow0 + r) = stats (aRow0 + r))
    (hround : roundBit6 = lnRound) :
    ∀ r k, r < M →
      prologueA A statsT roundBit6 x gamma beta aRow0 r k
        = layerNorm A stats lnRound x gamma beta (aRow0 + r) k := by
  intro r k hr
  simp [prologueA, layerNorm, hstats r hr, hround]

/-- Any GEMM that reads A only through its elements (whatever its accumulation order) gives
    the same output for the fused prologue and for the materialized LayerNormF32 output. -/
theorem ln_prologue_gemm_eq {α β : Type} (A : Arith α) (stats statsT : Nat → α × α)
    (lnRound roundBit6 : Bool) (x : Nat → Nat → α) (gamma beta : Nat → Option α)
    (aRow0 M : Nat) (hstats : ∀ r, r < M → statsT (aRow0 + r) = stats (aRow0 + r))
    (hround : roundBit6 = lnRound)
    (gemm : (Nat → Nat → α) → Nat → β) (hlocal : ∀ a a' : Nat → Nat → α,
      (∀ r k, r < M → a r k = a' r k) → ∀ n, gemm a n = gemm a' n) :
    ∀ n, gemm (prologueA A statsT roundBit6 x gamma beta aRow0) n
      = gemm (fun r k => layerNorm A stats lnRound x gamma beta (aRow0 + r) k) n :=
  hlocal _ _ (ln_prologue_eq A stats statsT lnRound roundBit6 x gamma beta aRow0 M hstats hround)

/-! ## Conv1dF32 row_scale residual -/

/-- Fused Conv1dF32 epilogue as `SpConvArgs::store` computes it:
    `__fadd_rn(__fmul_rn(row_scale, post(y)), residual)` (`y = bias + conv`). -/
def convFused {α : Type} (A : Arith α) (post : α → α) (y scale residual : α) : α :=
  A.add (A.mul scale (post y)) residual

/-- Unfused: Conv1dF32 writes `post(y)`, BinaryF32 mul by the row vector
    (`a * b`, a = conv output), BinaryF32 add with the residual as `a`. -/
def convUnfused {α : Type} (A : Arith α) (post : α → α) (y scale residual : α) : α :=
  A.add residual (A.mul (post y) scale)

theorem conv_row_scale_residual_eq {α : Type} (A : Arith α)
    (mulComm : ∀ a b, A.mul a b = A.mul b a) (addComm : ∀ a b, A.add a b = A.add b a)
    (post : α → α) (y scale residual : α) :
    convFused A post y scale residual = convUnfused A post y scale residual := by
  simp [convFused, convUnfused, mulComm (post y) scale, addComm residual]

end Plow.Speech
