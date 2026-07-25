/-
# Plow.CounterGranularity — WHEN are per-slice (fine) counters worth anything?

The interpreter gates every op on counters. The *granularity* of those counters is a
compiler choice:

* **coarse** — one counter per op. Every consumer workgroup waits for EVERY producer
  workgroup. This is what `devbuild` emitted for a long time.
* **fine** — one counter per producer slice. Consumer slice `v` waits only on the producer
  slices `P v` that actually feed it (a head, a column block, a KV split).

Fine counters look strictly better: a consumer stops waiting on producers it does not need.
They were built, and on the real Gemma-4 31B decode they were **worth exactly nothing** —
16.9 ms/token coarse, 17.2 ms fine — and no amount of widening the fine chain changed that.

This file says why, and turns the reason into a decision procedure the compiler can run.

## The result

`collapse` : if every stage's producer map COVERS the previous stage (the union of the `P v`
over all consumer slices is all of the producer's slices) and the work is UNIFORM across each
stage's slices, then the fine schedule's makespan is *identical* to the coarse one — for ANY
producer maps whatsoever.

The maps do not matter. The intuition is that a barrier at the end of the chain (in Gemma,
`o_proj`, which reduces over all heads) takes a `max` over all consumer slices, and

    max_v ( max_{u ∈ P v} finish u )  =  max_{u ∈ ⋃_v P v} finish u  =  max_{all u} finish u

so the global maximum is re-imposed no matter how finely the gates upstream were cut. Every
`max_v max_{P v}` collapses straight back to `max over everything`.

## The corollary the compiler needs

`collapse` is an *equality*, and its hypothesis is uniformity. So:

> **Fine counters can only pay when the per-slice work is NON-UNIFORM.**

`hetero_can_win` exhibits a two-slice witness where the fine makespan is strictly smaller —
the straggling producer feeds the *cheap* consumer, so its slack is absorbed instead of
propagating to the barrier. That is the entire opportunity, and it is why fine counters are
pointless for a transformer (every attention head does identical work) and are the right thing
for MoE (experts get different token counts by construction).

`fineCanPay` is the executable test: the compiler runs it on the cost model's per-slice work
and only emits fine counters where it returns `true`. On Gemma it returns `false` everywhere,
which is the correct answer and costs nothing to discover.
-/
import Plow.Basic

namespace Plow.CounterGranularity

/-! ## `maxOver` — the completion time of a set of slices. -/

/-- `max` of `f` over a list of slice ids. `0` on the empty list. -/
def maxOver (l : List Nat) (f : Nat → Nat) : Nat :=
  l.foldr (fun x acc => Nat.max (f x) acc) 0

@[simp] theorem maxOver_nil (f : Nat → Nat) : maxOver [] f = 0 := rfl

@[simp] theorem maxOver_cons (a : Nat) (l : List Nat) (f : Nat → Nat) :
    maxOver (a :: l) f = Nat.max (f a) (maxOver l f) := rfl

theorem le_maxOver {l : List Nat} {f : Nat → Nat} {u : Nat} (h : u ∈ l) :
    f u ≤ maxOver l f := by
  induction l with
  | nil => cases h
  | cons a t ih =>
    rcases List.mem_cons.mp h with rfl | ht
    · exact Nat.le_max_left _ _
    · exact Nat.le_trans (ih ht) (Nat.le_max_right _ _)

theorem maxOver_le {l : List Nat} {f : Nat → Nat} {b : Nat}
    (h : ∀ u ∈ l, f u ≤ b) : maxOver l f ≤ b := by
  induction l with
  | nil => exact Nat.zero_le _
  | cons a t ih =>
    exact Nat.max_le.mpr
      ⟨h a (List.mem_cons_self _ _),
       ih (fun u hu => h u (List.mem_cons_of_mem _ hu))⟩

/-- `max` commutes with a constant shift. Spelled out because `omega` will not split a
    `Nat.max` that sits under a `List.foldr`. -/
theorem max_add_right (x y c : Nat) : Nat.max (x + c) (y + c) = Nat.max x y + c := by
  simp only [Nat.max_def]
  split <;> split <;> omega

/-- Adding a constant to every slice's cost shifts the max — on a NON-EMPTY list.
    (On `[]` the max is `0`, not `0 + c`, which is exactly why the stages below are
    required to be non-empty: an op with no workgroups is not a thing.) -/
theorem maxOver_add (g : Nat → Nat) (c : Nat) :
    ∀ (a : Nat) (t : List Nat),
      maxOver (a :: t) (fun v => g v + c) = maxOver (a :: t) g + c := by
  intro a t
  induction t generalizing a with
  | nil => simp [maxOver]
  | cons b t ih =>
    -- unfold BOTH sides by `rfl` rather than `rw`: the shifted function is a lambda, and
    -- `rw`'s higher-order matching will not see through the beta redex reliably.
    have e1 : maxOver (a :: b :: t) (fun v => g v + c)
            = Nat.max (g a + c) (maxOver (b :: t) (fun v => g v + c)) := rfl
    have e2 : maxOver (a :: b :: t) g = Nat.max (g a) (maxOver (b :: t) g) := rfl
    rw [e1, e2, ih b, max_add_right]

/-! ## The key lemma: a covering map makes the max collapse. -/

/-- **The whole result in one line.** If the consumer slices' producer sets are sound (they
    only name real producers) and covering (between them they name every producer), then
    taking the max of per-consumer maxima is the same as taking the max over all producers.

    Note what is NOT assumed: nothing at all about the SHAPE of `P`. A fine map can be as
    sparse as you like — 8-of-256, 4-of-128 — and this still holds. Sparsity is not the
    point; coverage is. -/
theorem cover_max
    (cons : List Nat) (P : Nat → List Nat) (prod : List Nat) (f : Nat → Nat)
    (sound : ∀ v ∈ cons, ∀ u ∈ P v, u ∈ prod)
    (cover : ∀ u ∈ prod, ∃ v, v ∈ cons ∧ u ∈ P v) :
    maxOver cons (fun v => maxOver (P v) f) = maxOver prod f := by
  apply Nat.le_antisymm
  · exact maxOver_le (fun v hv =>
      maxOver_le (fun u hu => le_maxOver (sound v hv u hu)))
  · refine maxOver_le (fun u hu => ?_)
    rcases cover u hu with ⟨v, hv, huv⟩
    exact Nat.le_trans (le_maxOver huv) (le_maxOver (f := fun v => maxOver (P v) f) hv)

/-! ## A chain of counter-gated stages. -/

/-- One counter-gated op. `slices` are its workgroups, `P v` is the set of producer slices
    that consumer slice `v` waits on, and `w` is the work each slice does.

    `w : Nat` — a single number — IS the uniformity hypothesis, stated by construction. -/
structure Stage where
  slices : List Nat
  P      : Nat → List Nat
  w      : Nat

/-- Run a chain: each stage's finish function is `max over my producers, plus my work`. -/
def run (prevSlices : List Nat) (prevFin : Nat → Nat) : List Stage → List Nat × (Nat → Nat)
  | []      => (prevSlices, prevFin)
  | s :: rest => run s.slices (fun v => maxOver (s.P v) prevFin + s.w) rest

/-- Total work down the chain. -/
def totalWork : List Stage → Nat
  | []      => 0
  | s :: rest => s.w + totalWork rest

/-- The producer map of each stage is sound, covering, and the stage is non-empty. -/
inductive ChainCovers : List Nat → List Stage → Prop
  | nil {ps} : ChainCovers ps []
  | cons {ps s rest}
      (hne    : s.slices ≠ [])
      (sound  : ∀ v ∈ s.slices, ∀ u ∈ s.P v, u ∈ ps)
      (cover  : ∀ u ∈ ps, ∃ v, v ∈ s.slices ∧ u ∈ s.P v)
      (tail   : ChainCovers s.slices rest) :
      ChainCovers ps (s :: rest)

/-- **THE COLLAPSE THEOREM.**

    A chain of counter-gated stages, ending in a barrier (the `maxOver` on the left is the
    barrier: a reduction that consumes every slice of the last stage — `o_proj` in Gemma).

    Its makespan is `max over the ARRIVAL times` plus the total work — and it does not depend
    on the producer maps `P` in any way. So the fine schedule and the coarse schedule, which
    differ ONLY in `P`, have exactly the same makespan.

    Per-slice counters cannot recover the straggler. Its slack reaches the barrier regardless
    of who waits for whom. -/
theorem collapse (ps : List Nat) (f : Nat → Nat) :
    ∀ (chain : List Stage), ChainCovers ps chain →
      maxOver (run ps f chain).1 (run ps f chain).2 = maxOver ps f + totalWork chain := by
  intro chain
  induction chain generalizing ps f with
  | nil => intro _; simp [run, totalWork]
  | cons s rest ih =>
    intro hc
    cases hc with
    | cons hne sound cover tail =>
      -- one step: this stage's finish, maxed over its slices, is `maxOver ps f + s.w`
      have hstep :
          maxOver s.slices (fun v => maxOver (s.P v) f + s.w) = maxOver ps f + s.w := by
        have hcm := cover_max s.slices s.P ps f sound cover
        cases hsl : s.slices with
        | nil => exact absurd hsl hne
        | cons a t =>
          -- `cases hsl : s.slices` already substituted `a :: t` into the goal; only the
          -- previously-introduced `hcm` still mentions `s.slices`.
          rw [hsl] at hcm
          calc maxOver (a :: t) (fun v => maxOver (s.P v) f + s.w)
              = maxOver (a :: t) (fun v => maxOver (s.P v) f) + s.w :=
                maxOver_add _ _ a t
            _ = maxOver ps f + s.w := by rw [hcm]
      have hrest := ih (ps := s.slices) (f := fun v => maxOver (s.P v) f + s.w) tail
      simp only [run, totalWork]
      rw [hrest, hstep]
      omega

/-! ## And the corollary: only heterogeneity can win.

`collapse` is an EQUALITY whose hypothesis is that each stage's work is one number. Drop that
and fine counters can strictly win. Here is a witness, and it is the smallest honest one.

Two producers, `0` and `1`. Producer `0` straggles (finishes at 10); producer `1` is quick
(finishes at 0). Two consumers: consumer `0` waits only on producer `0` and is CHEAP (work 1);
consumer `1` waits only on producer `1` and is EXPENSIVE (work 8).

* coarse — every consumer waits for both producers, so both start at 10:
    finish = max(10 + 1, 10 + 8) = 18
* fine — each waits only on its own producer:
    finish = max(10 + 1, 0 + 8) = 11

The straggler's slack is ABSORBED by the cheap consumer instead of propagating to the barrier.
That is the entire opportunity, and it exists only because the two consumers do different
amounts of work.

In a transformer every attention head is identical, so this witness has no analogue and
`collapse` applies — which is precisely what the machine measured. In an MoE, experts get
different token counts by construction, and this witness IS the workload. -/

/-- Per-slice work (the thing `Stage.w` forbids). -/
abbrev Work := Nat → Nat

/-- Fine: consumer `v` starts after its own producers only. -/
def fineFinish (cons : List Nat) (P : Nat → List Nat) (prodFin : Nat → Nat) (w : Work) : Nat :=
  maxOver cons (fun v => maxOver (P v) prodFin + w v)

/-- Coarse: every consumer starts after ALL producers. -/
def coarseFinish (cons : List Nat) (prod : List Nat) (prodFin : Nat → Nat) (w : Work) : Nat :=
  maxOver cons (fun v => maxOver prod prodFin + w v)

/-- Fine is never WORSE: narrowing a wait set can only lower a start time. -/
theorem fine_le_coarse (cons : List Nat) (P : Nat → List Nat) (prod : List Nat)
    (prodFin : Nat → Nat) (w : Work)
    (sound : ∀ v ∈ cons, ∀ u ∈ P v, u ∈ prod) :
    fineFinish cons P prodFin w ≤ coarseFinish cons prod prodFin w := by
  refine maxOver_le (fun v hv => ?_)
  have : maxOver (P v) prodFin ≤ maxOver prod prodFin :=
    maxOver_le (fun u hu => le_maxOver (sound v hv u hu))
  exact Nat.le_trans (Nat.add_le_add_right this (w v))
    (le_maxOver (f := fun v => maxOver prod prodFin + w v) hv)

/-- …and with NON-UNIFORM work it can be strictly better: 11 < 18. -/
theorem hetero_can_win :
    fineFinish [0, 1] (fun v => [v]) (fun u => if u = 0 then 10 else 0)
        (fun v => if v = 0 then 1 else 8) <
      coarseFinish [0, 1] [0, 1] (fun u => if u = 0 then 10 else 0)
        (fun v => if v = 0 then 1 else 8) := by
  decide

/-! ## The executable decision the compiler runs. -/

/-- Is there anything for fine counters to win on this edge?

    `collapse` says: NO, unless the consumer slices do different amounts of work. So the
    compiler's test is not about the sparsity of the dependency map at all — it is about
    whether the cost model gives the consumer's slices differing costs.

    On Gemma this is `false` on every edge (all heads identical) and the compiler emits coarse
    counters, which is the measured optimum. On MoE it is `true`, and it should emit fine. -/
def fineCanPay (cons : List Nat) (w : Work) : Bool :=
  match cons with
  | []      => false
  | v :: vs => vs.any (fun u => w u != w v)

/-- Uniform work ⇒ the compiler declines fine counters. The contrapositive of `collapse`,
    in the form `plowc` actually calls. -/
theorem fineCanPay_false_of_uniform (cons : List Nat) (w : Work) (c : Nat)
    (hu : ∀ v ∈ cons, w v = c) : fineCanPay cons w = false := by
  cases cons with
  | nil => rfl
  | cons v vs =>
    have hv : w v = c := hu v (List.mem_cons_self _ _)
    have key : ∀ l : List Nat, (∀ u ∈ l, w u = c) →
        l.any (fun u => w u != w v) = false := by
      intro l
      induction l with
      | nil => intro _; rfl
      | cons a t ih =>
        intro h
        have ha : w a = c := h a (List.mem_cons_self _ _)
        have ht := ih (fun u hu' => h u (List.mem_cons_of_mem _ hu'))
        have hne : (w a != w v) = false := by rw [ha, hv]; exact bne_self_eq_false c
        simp [List.any_cons, hne, ht]
    exact key vs (fun u hu' => hu u (List.mem_cons_of_mem _ hu'))

end Plow.CounterGranularity
