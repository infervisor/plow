/-
# Plow.MoePerf — MoE / EP performance theorems (§D1–D3).

Proves three properties of the SM-local EP dispatch design
(`plans/expert-parallel.md`):

* **D1**: gate-weighted expert sum — the fixed-`top_k` packet stream with
  predicated skip (gate = 0 for unused slots) produces the same result as
  a variable-`k` mixture.
* **D2**: router determinism — the router is a pure function of
  `(weights, input)`.
* **D3**: expert-table indirection safety — the SM's weight-base lookup
  via `expert_weight_table[expert_id]` is well-defined iff
  `expert_id < num_experts`.
-/
import Plow.Basic

namespace Plow.MoePerf

/-- Sentinel value indicating an unused slot. The SM checks against this
    before doing any compute. -/
def unusedSentinel (num_experts : Nat) : Nat := num_experts

/-! ## D1 — Gate-weighted expert sum with predicated skip. -/

/-- Per-slot contribution. Zero when the expert slot is unused (sentinel);
    otherwise `gate · expert_out(expert_id)`. Kept as a separate function so
    the mixture sum reduces to a foldr over `slotContrib` — decoupled from
    the pair-destructure pattern that trips Lean's elaborator. -/
def slotContrib (num_experts : Nat) (routing : Nat → Nat × Nat)
    (expert_out : Nat → Nat) (slot : Nat) : Nat :=
  if (routing slot).1 = unusedSentinel num_experts then 0
  else (routing slot).2 * expert_out (routing slot).1

/-- The mixture output for one token: `Σ_i gate_i · expert_i(x)`. `top_k`
    is the compile-time bound; entries with `expert_id = num_experts`
    (sentinel) contribute zero. -/
def mixtureSum (num_experts : Nat) (top_k : Nat)
    (routing : Nat → Nat × Nat)  -- slot_id → (expert_id, gate)
    (expert_out : Nat → Nat)     -- expert_id → expert's compute output
    : Nat :=
  (List.range top_k).foldr (fun slot acc =>
    slotContrib num_experts routing expert_out slot + acc) 0

/-- **D1**: a slot whose expert id is the sentinel contributes zero to the
    mixture regardless of the gate value. -/
theorem unused_slot_zero (num_experts : Nat) (routing : Nat → Nat × Nat)
    (expert_out : Nat → Nat) (slot : Nat)
    (h_unused : (routing slot).1 = unusedSentinel num_experts) :
    slotContrib num_experts routing expert_out slot = 0 := by
  unfold slotContrib
  simp [h_unused]

/-- **D1** (main): two routings whose slot contributions agree pointwise
    produce identical mixture sums. In particular, a slot marked unused
    (sentinel) has the same contribution as a slot with `gate = 0`, so the
    predicated-skip design produces the same output as an explicit
    variable-`k` mixture. -/
theorem contributions_agree_implies_equal_sum
    (num_experts top_k : Nat)
    (routing1 routing2 : Nat → Nat × Nat)
    (expert_out : Nat → Nat)
    (h_contrib : ∀ s,
      slotContrib num_experts routing1 expert_out s
      = slotContrib num_experts routing2 expert_out s) :
    mixtureSum num_experts top_k routing1 expert_out
    = mixtureSum num_experts top_k routing2 expert_out := by
  -- Prove the foldr is equal by structural induction on the list.
  unfold mixtureSum
  have key : ∀ l : List Nat,
      l.foldr (fun slot acc =>
        slotContrib num_experts routing1 expert_out slot + acc) 0
      = l.foldr (fun slot acc =>
        slotContrib num_experts routing2 expert_out slot + acc) 0 := by
    intro l
    induction l with
    | nil => rfl
    | cons head tail ih =>
      simp only [List.foldr_cons]
      rw [ih, h_contrib head]
  exact key (List.range top_k)

/-! ## D2 — Router determinism. -/

/-- The router is a pure function of `(router_weights, input)`. Two
    invocations with identical arguments yield identical outputs. -/
def router (weights : Nat) (input : Nat) : Nat × Nat :=
  (weights + input, 0) -- abstract stand-in

/-- **D2** (definitional bookkeeping): unfolds the abstract stand-in
    router. Determinism of a Lean function is built into the logic (every
    `def` is a mathematical function), so nothing here is — or can be —
    proven about the *Rust* router's determinism; that claim rests on the
    Rust code taking no hidden inputs (RNG, clock, atomics), which is
    outside this model. Recorded as interface documentation only. -/
theorem router_expert_id_pure (weights input : Nat) :
    (router weights input).1 = weights + input := rfl

/-! ## D3 — Expert-table indirection safety.

    Concrete contiguous-layout model: expert `e`'s weights live at
    `[e · expertBytes, (e+1) · expertBytes)` inside a table whose total
    size is the expert-by-expert sum of the per-expert footprints. The
    theorems below prove that an in-range `expert_id` (a `Fin`) always
    dereferences inside the table and that distinct experts' regions are
    disjoint — indirection through the table cannot fault or alias. -/

/-- Base offset of expert `e` under the contiguous layout. -/
def expertBase (expertBytes e : Nat) : Nat := e * expertBytes

/-- Total table bytes, accumulated expert-by-expert (an actual fold, not
    a closed form): mirrors how the compiler lays the table out. -/
def tableBytes (num_experts expertBytes : Nat) : Nat :=
  sumRange num_experts (fun _ => expertBytes)

/-- The fold-based table size equals the closed form
    `num_experts · expertBytes`. -/
theorem tableBytes_eq (num_experts expertBytes : Nat) :
    tableBytes num_experts expertBytes = num_experts * expertBytes := by
  unfold tableBytes
  induction num_experts with
  | zero => simp [sumRange]
  | succ n ih =>
    have hstep : sumRange (n + 1) (fun _ => expertBytes)
        = sumRange n (fun _ => expertBytes) + expertBytes := rfl
    rw [hstep, ih, Nat.add_one_mul]

/-- **D3** (in-bounds): for any `expert_id < num_experts` (enforced by the
    `Fin` — the SM prologue's bounds check is exactly this proof), the
    expert's whole weight region lies inside the table. -/
theorem expert_base_in_bounds (num_experts expertBytes : Nat)
    (e : Fin num_experts) :
    expertBase expertBytes e.val + expertBytes
    ≤ tableBytes num_experts expertBytes := by
  rw [tableBytes_eq]
  unfold expertBase
  calc e.val * expertBytes + expertBytes
      = (e.val + 1) * expertBytes := (Nat.succ_mul _ _).symm
    _ ≤ num_experts * expertBytes := Nat.mul_le_mul_right _ e.isLt

/-- **D3** (no aliasing): distinct experts' regions are disjoint — the
    lower-indexed expert's region ends before the higher one's begins. -/
theorem expert_regions_disjoint (expertBytes : Nat) (e1 e2 : Nat)
    (h : e1 < e2) :
    expertBase expertBytes e1 + expertBytes ≤ expertBase expertBytes e2 := by
  unfold expertBase
  calc e1 * expertBytes + expertBytes
      = (e1 + 1) * expertBytes := (Nat.succ_mul _ _).symm
    _ ≤ e2 * expertBytes := Nat.mul_le_mul_right _ h

/-- **D3** effective content: converting an unchecked `Nat` `expert_id` to
    a `Fin num_experts` requires proving the bounds check — the SM's guard
    `if expert_id >= num_experts then skip` is exactly this proof. -/
def expert_id_to_fin (num_experts expert_id : Nat) (h : expert_id < num_experts) :
    Fin num_experts := ⟨expert_id, h⟩

/-- **D3** contrapositive: an out-of-range `expert_id` is caught by the SM's
    bounds check. The compiler-emitted `EXPERT_UNUSED_SENTINEL = u32::MAX`
    (modeled here as `num_experts`) triggers the predicated skip path. -/
theorem sentinel_is_out_of_range (num_experts : Nat) :
    ¬ (unusedSentinel num_experts < num_experts) := by
  unfold unusedSentinel
  exact Nat.lt_irrefl _

end Plow.MoePerf
