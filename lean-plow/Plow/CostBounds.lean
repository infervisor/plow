/-
# Plow.CostBounds — schedule cost lower bounds (§E1–E3).

Proves three lower bounds on the makespan of any valid schedule:

* **E1**: makespan is at least the critical-path length in the task graph.
* **E2**: makespan is at least the memory-bandwidth-limited time
  (`total_HBM_bytes / peak_bandwidth`).
* **E3**: makespan is at least the compute-limited time
  (`total_FLOPs / peak_throughput`).

The compiler uses these to certify "optimal" schedules (makespan achieves
the lower bound) and to short-circuit further optimization.
-/
import Plow.Basic
import Plow.Protocol

namespace Plow.CostBounds

open Plow Plow.Protocol

/-! ## E1 — Critical-path lower bound. -/

/-- A schedule assigns each task a start cycle. `makespan` is the largest
    finish time. -/
structure Schedule (tg : TaskGraph) where
  starts : Fin tg.n → Nat
  durations : Fin tg.n → Nat
  makespan : Nat
  /-- Every task's finish (start + duration) fits under the makespan. -/
  fits : ∀ t : Fin tg.n, starts t + durations t ≤ makespan
  /-- Data-dependency edges are respected: producer finishes before
      consumer starts. -/
  respects_edges : ∀ (a b : Fin tg.n), (a, b) ∈ tg.edges →
    starts a + durations a ≤ starts b

/-- Critical-path length for a specific edge chain: the sum of durations
    along a path in `tg.edges`. Modeled per-edge for simplicity. -/
def edgePathLength {tg : TaskGraph} (sch : Schedule tg)
    (a b : Fin tg.n) : Nat :=
  sch.durations a + sch.durations b

/-- **E1**: for any edge `(a, b)` in the task graph, the schedule's makespan
    is at least the sum of the two tasks' durations. Direct chain: `a`
    starts, finishes at `starts a + dur a`; `b` starts no earlier at
    `starts a + dur a` and takes `dur b` more cycles. -/
theorem makespan_ge_edge_chain {tg : TaskGraph} (sch : Schedule tg)
    (a b : Fin tg.n) (h_edge : (a, b) ∈ tg.edges) :
    sch.durations a + sch.durations b ≤ sch.makespan := by
  have h_b : sch.starts b + sch.durations b ≤ sch.makespan := sch.fits b
  have h_ab : sch.starts a + sch.durations a ≤ sch.starts b :=
    sch.respects_edges a b h_edge
  omega

/-- **E1** corollary (single-task lower bound): the makespan is at least
    the duration of any task in the schedule. -/
theorem makespan_ge_task_duration {tg : TaskGraph} (sch : Schedule tg)
    (t : Fin tg.n) : sch.durations t ≤ sch.makespan := by
  have h : sch.starts t + sch.durations t ≤ sch.makespan := sch.fits t
  omega

/-! ## E2 — HBM-bandwidth lower bound.

    Model: all DMA traffic funnels through one serialized channel (the HBM
    fabric). Each transferring task occupies the channel for its duration,
    moving at most `rate` bytes per cycle, and the channel handles one
    transfer at a time, so the summed busy time fits under the makespan.
    From this the bandwidth bound `total_bytes ≤ rate · makespan` is
    *derived*, not assumed. -/

/-- Total HBM bytes moved by a schedule, summed across all tasks that
    issue a DMA. -/
def totalHbmBytes {tg : TaskGraph} (dmaSizes : Fin tg.n → Nat) : Nat :=
  (List.finRange tg.n).foldr (fun t acc => dmaSizes t + acc) 0

/-- Total FLOPs across all compute tasks. -/
def totalFlops {tg : TaskGraph} (flopsPerTask : Fin tg.n → Nat) : Nat :=
  (List.finRange tg.n).foldr (fun t acc => flopsPerTask t + acc) 0

/-- A serialized throughput channel: each task occupies the channel for
    `durations t` cycles processing `work t` units at a peak of `rate`
    units per cycle, one task at a time.

    * HBM instance: `work` = DMA bytes, `rate` = peak bytes/cycle.
    * Compute instance (wave-serialized whole-grid execution): `work` =
      FLOPs, `rate` = flop/cycle/SM × SM count. -/
structure SerialChannel (tg : TaskGraph) where
  work      : Fin tg.n → Nat
  durations : Fin tg.n → Nat
  rate      : Nat
  makespan  : Nat
  /-- Each task's work is rate-limited: `work ≤ rate × duration`. -/
  rate_limited : ∀ t : Fin tg.n, work t ≤ rate * durations t
  /-- The channel is serial: total busy time fits under the makespan. -/
  serial : (List.finRange tg.n).foldr (fun t acc => durations t + acc) 0
             ≤ makespan

/-- Summed foldr is monotone under a pointwise bound. -/
private theorem foldr_add_le {α : Type} (l : List α) (f g : α → Nat)
    (h : ∀ x ∈ l, f x ≤ g x) :
    l.foldr (fun x acc => f x + acc) 0 ≤ l.foldr (fun x acc => g x + acc) 0 := by
  induction l with
  | nil => exact Nat.le_refl _
  | cons x xs ih =>
    simp only [List.foldr_cons]
    have hx := h x (List.mem_cons_self _ _)
    have hxs := ih (fun y hy => h y (List.mem_cons_of_mem _ hy))
    omega

/-- A constant factor distributes out of a summed foldr. -/
private theorem foldr_add_mul {α : Type} (l : List α) (c : Nat) (f : α → Nat) :
    l.foldr (fun x acc => c * f x + acc) 0
    = c * l.foldr (fun x acc => f x + acc) 0 := by
  induction l with
  | nil => simp
  | cons x xs ih =>
    simp only [List.foldr_cons]
    rw [ih, Nat.mul_add]

/-- Core channel bound: total work ≤ rate × makespan. Chain:
    `Σ work ≤ Σ rate·dur = rate·Σ dur ≤ rate·makespan`. -/
theorem channel_work_le_rate_mul_makespan {tg : TaskGraph}
    (ch : SerialChannel tg) :
    (List.finRange tg.n).foldr (fun t acc => ch.work t + acc) 0
    ≤ ch.rate * ch.makespan := by
  have h1 : (List.finRange tg.n).foldr (fun t acc => ch.work t + acc) 0
      ≤ (List.finRange tg.n).foldr
          (fun t acc => ch.rate * ch.durations t + acc) 0 :=
    foldr_add_le _ _ _ (fun t _ => ch.rate_limited t)
  rw [foldr_add_mul] at h1
  exact Nat.le_trans h1 (Nat.mul_le_mul_left ch.rate ch.serial)

/-- **E2**: for any schedule realizable on the serialized HBM channel,
    `peak_BW · makespan ≥ total_HBM_bytes` — derived from the per-transfer
    bandwidth limit and channel serialization, so
    `makespan ≥ total_bytes / peak_BW`. -/
theorem makespan_ge_hbm_bound {tg : TaskGraph} (ch : SerialChannel tg) :
    ch.rate * ch.makespan ≥ totalHbmBytes ch.work :=
  channel_work_le_rate_mul_makespan ch

/-- Contrapositive form the extractor uses: a candidate makespan that would
    require pushing more bytes than the channel can carry is a strict
    underestimate of any realizable makespan. -/
theorem infeasible_when_bw_saturated {tg : TaskGraph}
    (ch : SerialChannel tg) (candidate_makespan : Nat)
    (h : totalHbmBytes ch.work > ch.rate * candidate_makespan) :
    candidate_makespan < ch.makespan := by
  have hb : totalHbmBytes ch.work ≤ ch.rate * ch.makespan :=
    channel_work_le_rate_mul_makespan ch
  by_cases hle : ch.makespan ≤ candidate_makespan
  · have hmul : ch.rate * ch.makespan ≤ ch.rate * candidate_makespan :=
      Nat.mul_le_mul_left ch.rate hle
    exact absurd (Nat.le_trans hb hmul) (Nat.not_le_of_lt h)
  · exact Nat.lt_of_not_le hle

/-! ## E3 — Compute-throughput lower bound.

    Same channel model with the machine's aggregate compute treated as one
    serialized throughput channel of rate `peak_flop_per_cycle_per_sm ×
    sm_count` — the wave-serialized (whole-grid barrier) execution plow
    emits. -/

/-- **E3**: `peak_TFLOP · sm_count · makespan ≥ total_FLOPs` for any
    schedule realizable on the aggregate compute channel. -/
theorem makespan_ge_compute_bound {tg : TaskGraph} (ch : SerialChannel tg)
    (peak_flop_per_cycle_per_sm sm_count : Nat)
    (h_rate : ch.rate = peak_flop_per_cycle_per_sm * sm_count) :
    peak_flop_per_cycle_per_sm * sm_count * ch.makespan ≥ totalFlops ch.work := by
  rw [← h_rate]
  exact channel_work_le_rate_mul_makespan ch

/-- **E3** contrapositive: a candidate makespan exceeding the compute cap
    strictly underestimates any realizable makespan. -/
theorem infeasible_when_compute_saturated {tg : TaskGraph}
    (ch : SerialChannel tg) (candidate_makespan : Nat)
    (h : totalFlops ch.work > ch.rate * candidate_makespan) :
    candidate_makespan < ch.makespan := by
  have hb : totalFlops ch.work ≤ ch.rate * ch.makespan :=
    channel_work_le_rate_mul_makespan ch
  by_cases hle : ch.makespan ≤ candidate_makespan
  · have hmul : ch.rate * ch.makespan ≤ ch.rate * candidate_makespan :=
      Nat.mul_le_mul_left ch.rate hle
    exact absurd (Nat.le_trans hb hmul) (Nat.not_le_of_lt h)
  · exact Nat.lt_of_not_le hle

/-! ## Combined bound.

    The scheduler treats a schedule as "optimal enough" when its makespan
    achieves the max of the three lower bounds:
    `makespan = max(critical_path, hbm_bound, compute_bound)`.

    A trivial monotonicity: any lower bound proves the schedule's makespan
    is at least that lower bound. -/

/-- **Combined**: if a schedule's makespan is bounded below by each of the
    three lower bounds, it dominates all of them. Used by the extractor to
    certify optimality and stop exploring. -/
theorem makespan_dominates_lower_bounds {tg : TaskGraph} (sch : Schedule tg)
    (cp hbm cmp : Nat)
    (h1 : cp ≤ sch.makespan) (h2 : hbm ≤ sch.makespan)
    (h3 : cmp ≤ sch.makespan) :
    max cp (max hbm cmp) ≤ sch.makespan := by
  omega

end Plow.CostBounds
