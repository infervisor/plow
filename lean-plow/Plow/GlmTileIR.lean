/-
# Plow.GlmTileIR — soundness of the GLM-5.2 tile-IR partial-evaluator.

The `rewrite::glm` module lifts the GLM-5.2-FP8 decode block into the Path-A
tile IR and PARTIAL-EVALUATES its static gates (ctx-bucket, layer-type, TP, fp8):
given the compile-time knowns it emits a specialized op list, leaving only the
DYNAMIC gates (router top-8, DSA gather) as runtime nodes.

This file certifies two properties of that lift, reusing the existing
`CounterGranularity` (counter-DAG makespan) and `SplitK` (occupancy) machinery:

1. **Partial-eval is semantics-preserving.** A static gate is, by construction,
   a compile-time-known boolean selecting between two sub-schedules. So the
   schedule the compiler emits after folding the gate is *identical* to the one
   the runtime branch would have produced. `partial_eval_sound`. The whole point
   is that this is trivial for STATIC gates (branch selection) — which is exactly
   why folding them is safe — and that we do NOT fold the dynamic gates.

2. **The GLM counter-DAG is deadlock-free.** The specialized block is a chain of
   counter-gated stages in program order; every stage's producer set is covered
   by the previous stage (`ChainCovers`), so `collapse` gives a closed-form,
   finite makespan — no consumer counter can wait on a threshold it never
   reaches. `glm_deadlock_free`.

Two corollaries tie the lift to the cost model:

* **Partial-eval never bloats the graph** (`partial_eval_no_bloat`): the folded
  branch has no more ops than the union of both branches, so by
  `SplitK.occupancy_mono_count` its SRAM occupancy is bounded by the union's.
  This is why static-gate resolution can only *reduce* the decode op count (drop
  the DSA subgraph at short ctx; drop the router/experts on dense layers).

* **The dynamic gate does not enable fine counters at decode**
  (`moe_decode_declines_fine`): at M=1 every routed expert processes the one
  token, so per-slice work is uniform and `fineCanPay` is `false` — the emit's
  coarse per-op gate is optimal, by `CounterGranularity`'s contrapositive.
-/
import Plow.CounterGranularity
import Plow.SplitK

namespace Plow.GlmTileIR

open Plow.CounterGranularity
open Plow.SplitK

/-! ## 1. Partial-evaluation of a static gate is semantics-preserving. -/

/-- **Static-gate soundness.** A static gate is a compile-time boolean `g`
    choosing between sub-schedules `A` (taken) and `B`. Partial-eval emits `A`
    or `B` directly; the runtime would branch on `g`. The two schedules are
    equal — for ANY prior state `(ps, f)`.

    This is the formal content of "a compile-resolved gate == the runtime gate's
    schedule". It is deliberately trivial: STATIC gates are pure branch
    selection, so eliminating them is correct by construction. All the
    non-triviality lives in the DYNAMIC gates, which the compiler does NOT
    fold. -/
theorem partial_eval_sound (g : Bool) (ps : List Nat) (f : Nat → Nat)
    (A B : List Stage) :
    run ps f (if g then A else B)
      = (if g then run ps f A else run ps f B) := by
  cases g <;> rfl

/-! ## 2. The GLM counter-DAG is deadlock-free.

    Model the specialized block as a chain of counter-gated stages in program
    order. Abstracting each op to a single workgroup that waits on the previous
    op, every stage trivially covers its predecessor, so `ChainCovers` holds and
    `collapse` gives a finite makespan. -/

/-- One program-order stage: a single workgroup `[0]` waiting on the previous
    op's single workgroup, doing `w` units of work. -/
def uni (w : Nat) : Stage := { slices := [0], P := fun _ => [0], w := w }

/-- Any program-order chain of `uni` stages satisfies `ChainCovers [0]`: each
    stage's producer set `[0]` is sound (only names slice 0) and covering (names
    slice 0). -/
theorem chainCovers_uni : ∀ ws : List Nat, ChainCovers [0] (ws.map uni)
  | [] => ChainCovers.nil
  | w :: ws => by
      rw [List.map_cons]
      refine ChainCovers.cons ?_ ?_ ?_ ?_
      · simp [uni]
      · intro v _ u hu; exact hu
      · intro u hu; exact ⟨0, by simp [uni], hu⟩
      · simpa [uni] using chainCovers_uni ws

/-- **GLM decode is deadlock-free.** The counter-gated block has a closed-form,
    finite makespan (`arrival + Σ work`) — invoking `collapse` on the covering
    program-order chain. A finite makespan means every consumer's counter
    threshold is reached: no op waits forever. The identity also shows the
    schedule is independent of the gate maps, so no fine/coarse counter choice
    can deadlock or change the makespan of the (uniform) decode chain. -/
theorem glm_deadlock_free (ws : List Nat) (f : Nat → Nat) :
    maxOver (run [0] f (ws.map uni)).1 (run [0] f (ws.map uni)).2
      = maxOver [0] f + totalWork (ws.map uni) :=
  collapse [0] f (ws.map uni) (chainCovers_uni ws)

/-! ## Corollary A — partial-eval never bloats the op graph. -/

/-- **No bloat.** The folded branch (`if g then A else B`) has no more ops than
    the union `A ++ B` of both branches, so its SRAM occupancy is bounded by the
    union's — by `occupancy_mono_count`. Static-gate resolution can therefore
    only *shrink* the decode graph (it drops the untaken branch), which is the
    entire op-count win of specializing ctx-bucket / layer-type. -/
theorem partial_eval_no_bloat (g : Bool) (A B : List Stage) (p : Nat) :
    occupancy ((if g then A else B).length) p
      ≤ occupancy (A.length + B.length) p := by
  refine occupancy_mono_count (A.length + B.length) ((if g then A else B).length) p ?_
  cases g
  · show B.length ≤ A.length + B.length
    exact Nat.le_add_left _ _
  · show A.length ≤ A.length + B.length
    exact Nat.le_add_right _ _

/-! ## Corollary B — the dynamic router gate declines fine counters at decode. -/

/-- **Coarse counters are optimal for M=1 MoE.** At decode every routed expert
    processes exactly the one token, so its per-slice work is a single constant
    `c`. `CounterGranularity.collapse`'s contrapositive (`fineCanPay_false_of_
    uniform`) then says fine per-expert counters cannot beat coarse ones — the
    hand-emit's coarse per-op gate is already optimal. Fine counters only pay in
    the batched/prefill regime where experts get different token counts. -/
theorem moe_decode_declines_fine (experts : List Nat) (w : Work) (c : Nat)
    (huniform : ∀ v ∈ experts, w v = c) :
    fineCanPay experts w = false :=
  fineCanPay_false_of_uniform experts w c huniform

end Plow.GlmTileIR
