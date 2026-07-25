/-
# Plow.MlpInterleave — the interleaved gate/up weight layout is a SAFE relabelling.

The MLP gate/up block computes, for input `x` and weight rows `Wg`, `Wu`:

    gate[m,n] = <x[m], Wg[n]>,   up[m,n] = <x[m], Wu[n]>,   out[m,n] = act(gate)·up.

plow's `d_gemm_glu` runs ONE fused GEMM over both projections and applies the GLU in the
epilogue. There are three ways to lay the `2N` weight rows in memory:

* **split**    — two tensors `Wg`, `Wu` (plow today; a branch per B-row).
* **contig**   — one `[2N,K]` tensor, gate rows `[0,N)` then up rows `[N,2N)` (single pointer, +N offset).
* **interleave** — one `[2N,K]` tensor, rows alternate `2n = Wg[n]`, `2n+1 = Wu[n]`.

The proposed experiment changes the STORED layout, so before measuring we prove the change is
correct: **all three layouts compute the same MLP output**, the interleave is a bijection on rows
(no row dropped or duplicated), and the de-interleave B-tile address map the kernel uses is a
bijection that sends the tile's low half to gate rows and its high half to up rows — the exact
arrangement `d_gemm_glu`'s epilogue (gate in the SN=0 slice, up in SN=1) requires.

The values (`dot`, `act`, `mul`) are OPAQUE: the theorems are about the index algebra, so they
hold over any element type and any activation — the layout carries no arithmetic, only addresses.
-/

namespace Plow.MlpInterleave

section
variable {α : Type}
variable (dot : (Nat → α) → (Nat → α) → α) (act : α → α) (mul : α → α → α)
variable (x Wg Wu : Nat → Nat → α)

/-- `gate[m,n] = <x[m], Wg[n]>`. -/
def gate (m n : Nat) : α := dot (x m) (Wg n)
/-- `up[m,n] = <x[m], Wu[n]>`. -/
def up (m n : Nat) : α := dot (x m) (Wu n)
/-- Reference GLU MLP output: `act(gate) · up`. -/
def outRef (m n : Nat) : α := mul (act (gate dot x Wg m n)) (up dot x Wu m n)

/-! ## Interleaved layout: row `2n` is gate[n], row `2n+1` is up[n]. -/

def Wfused (j : Nat) : Nat → α := if j % 2 = 0 then Wg (j / 2) else Wu (j / 2)

/-- Fused GEMM column `j` under the interleaved layout. -/
def fusedCol (m j : Nat) : α := dot (x m) (Wfused Wg Wu j)

/-- Interleaved MLP: pair the adjacent columns `2n` (gate) and `2n+1` (up). -/
def outInt (m n : Nat) : α :=
  mul (act (fusedCol dot x Wg Wu m (2 * n))) (fusedCol dot x Wg Wu m (2 * n + 1))

/-- Even columns of the fused GEMM are exactly the gate projection. -/
theorem fusedCol_even (m n : Nat) : fusedCol dot x Wg Wu m (2 * n) = gate dot x Wg m n := by
  unfold fusedCol Wfused gate
  have hc : (2 * n) % 2 = 0 := by omega
  have hd : (2 * n) / 2 = n := by omega
  rw [if_pos hc, hd]

/-- Odd columns of the fused GEMM are exactly the up projection. -/
theorem fusedCol_odd (m n : Nat) : fusedCol dot x Wg Wu m (2 * n + 1) = up dot x Wu m n := by
  unfold fusedCol Wfused up
  have hc : ¬ ((2 * n + 1) % 2 = 0) := by omega
  have hd : (2 * n + 1) / 2 = n := by omega
  rw [if_neg hc, hd]

/-- **Main equivalence**: the interleaved layout computes the reference MLP, for any activation. -/
theorem interleave_correct (m n : Nat) :
    outInt dot act mul x Wg Wu m n = outRef dot act mul x Wg Wu m n := by
  unfold outInt outRef
  rw [fusedCol_even, fusedCol_odd]

/-! ## Contiguous layout: gate block `[0,N)` then up block `[N,2N)`. -/

def Wcontig (N j : Nat) : Nat → α := if j < N then Wg j else Wu (j - N)

def outContig (N m n : Nat) : α :=
  mul (act (dot (x m) (Wcontig Wg Wu N n))) (dot (x m) (Wcontig Wg Wu N (N + n)))

/-- **The contiguous layout also computes the reference MLP** (needs `n < N`). -/
theorem contig_correct (N m n : Nat) (hn : n < N) :
    outContig dot act mul x Wg Wu N m n = outRef dot act mul x Wg Wu m n := by
  unfold outContig outRef gate up Wcontig
  have h1 : (n < N) := hn
  have h2 : ¬ (N + n < N) := by omega
  have h3 : (N + n) - N = n := by omega
  rw [if_pos h1, if_neg h2, h3]

end

/-! ## The row layout is a bijection — no weight row is dropped or duplicated.

`φ (n, s) = 2n + s` (s = 0 gate, s = 1 up) with inverse `ψ j = (j/2, j%2)`. -/

/-- Interleave a `(column, side)` pair to a fused row. `side` : 0 = gate, 1 = up. -/
def ilv (n side : Nat) : Nat := 2 * n + side
/-- De-interleave a fused row to `(column, side)`. -/
def dilv (j : Nat) : Nat × Nat := (j / 2, j % 2)

/-- `ilv` then `dilv` is the identity on valid `(n, side<2)` pairs. -/
theorem dilv_ilv (n side : Nat) (hs : side < 2) : dilv (ilv n side) = (n, side) := by
  unfold dilv ilv
  have : (2 * n + side) / 2 = n := by omega
  have : (2 * n + side) % 2 = side := by omega
  simp_all

/-- `dilv` then `ilv` is the identity on every fused row `j`. -/
theorem ilv_dilv (j : Nat) : ilv (dilv j).1 (dilv j).2 = j := by
  unfold dilv ilv; omega

/-- The side is always `< 2`, so `dilv` lands in a legal `(column, side)`. -/
theorem dilv_side_lt (j : Nat) : (dilv j).2 < 2 := by unfold dilv; omega

/-! ## The kernel's de-interleave B-tile address map is correct.

For a tile whose output columns are `[n0, n0+H)` (`H = BN/2`), the B-tile of `2H` rows is
loaded from `W_fused` by `srcRow n0 H br`. The kernel needs: the LOW half `br ∈ [0,H)` to hold
the GATE rows of those columns, the HIGH half `br ∈ [H,2H)` to hold the UP rows. -/

/-- `src_row = 2·n0 + 2·(br % H) + [br ≥ H]`  (plow's B-tile de-interleave). -/
def srcRow (n0 H br : Nat) : Nat := 2 * n0 + 2 * (br % H) + (if br < H then 0 else 1)

/-- Low half of the tile ⇒ the GATE row (`2·col`) of column `n0 + br`. -/
theorem srcRow_lo (n0 H br : Nat) (h : br < H) :
    srcRow n0 H br = ilv (n0 + br) 0 := by
  unfold srcRow ilv
  have hb : br % H = br := Nat.mod_eq_of_lt h
  rw [hb, if_pos h]; omega

/-- High half of the tile ⇒ the UP row (`2·col + 1`) of column `n0 + (br − H)`. -/
theorem srcRow_hi (n0 H br : Nat) (hlo : H ≤ br) (hhi : br < 2 * H) :
    srcRow n0 H br = ilv (n0 + (br - H)) 1 := by
  unfold srcRow ilv
  have hlt : br - H < H := by omega
  have hb : br % H = br - H := by
    rw [Nat.mod_eq_sub_mod hlo, Nat.mod_eq_of_lt hlt]
  have hnlt : ¬ (br < H) := by omega
  rw [hb, if_neg hnlt]; omega

/-- The de-interleave map is a **bijection** from tile rows `[0, 2H)` onto the fused rows of the
    columns `[n0, n0+H)`: it is injective (no two tile rows read the same weight row). -/
theorem srcRow_injective (n0 H : Nat) (hH : 0 < H)
    {a b : Nat} (ha : a < 2 * H) (hb : b < 2 * H)
    (heq : srcRow n0 H a = srcRow n0 H b) : a = b := by
  unfold srcRow at heq
  have modhi : ∀ c, H ≤ c → c < 2 * H → c % H = c - H := fun c h1 h2 => by
    rw [Nat.mod_eq_sub_mod h1, Nat.mod_eq_of_lt (by omega)]
  -- split each of a,b by the low/high half; the `%H` and the `[·≥H]` bit pin them down.
  rcases Nat.lt_or_ge a H with haH | haH <;> rcases Nat.lt_or_ge b H with hbH | hbH
  · rw [Nat.mod_eq_of_lt haH, Nat.mod_eq_of_lt hbH, if_pos haH, if_pos hbH] at heq; omega
  · rw [Nat.mod_eq_of_lt haH, modhi b hbH hb, if_pos haH, if_neg (by omega)] at heq; omega
  · rw [modhi a haH ha, Nat.mod_eq_of_lt hbH, if_neg (by omega), if_pos hbH] at heq; omega
  · rw [modhi a haH ha, modhi b hbH hb, if_neg (by omega), if_neg (by omega)] at heq; omega

end Plow.MlpInterleave
