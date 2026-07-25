/-
# Plow.Protocol — counter-protocol universal lemmas.

Encodes the abstract counter protocol the Rust scheduler emits, and the
happens-before relation it induces on tasks. The main theorem
`protocol_covers_deps` (§2 of the formal-verification plan) states that
the protocol enforces every data-dependency edge in the task graph.
-/
import Plow.Basic

namespace Plow.Protocol

open Plow

/-- A simplified task-graph view: tasks (by id) with their data-dependency
    edges. `n` is the number of tasks; edges are pairs over `Fin n`. -/
structure TaskGraph where
  n     : Nat
  edges : List (Fin n × Fin n)

/-- The counter protocol as the scheduler hands it to the runtime. -/
structure CounterProtocol (tg : TaskGraph) where
  /-- For each task, the set of counters it must wait on before issuing. -/
  waits     : Fin tg.n → List CounterId
  /-- For each task, the set of counters it increments on completion. -/
  succs     : Fin tg.n → List CounterId
  /-- The "all-complete" threshold of a counter (= number of producers). -/
  threshold : CounterId → Nat
  /-- Resource a task is placed on. Same-resource tasks issue in stream order. -/
  resource  : Fin tg.n → ResourceId
  /-- Stream index of a task within its resource. Lower = earlier issued. -/
  streamIdx : Fin tg.n → Nat

/-- Counter-gated edge: a increments c, b waits on c. -/
def counterGated {tg : TaskGraph} (p : CounterProtocol tg) (a b : Fin tg.n) : Prop :=
  ∃ c : CounterId, c ∈ p.succs a ∧ c ∈ p.waits b

/-- Resource-ordered edge: same resource, a issues before b. -/
def resourceOrdered {tg : TaskGraph} (p : CounterProtocol tg) (a b : Fin tg.n) : Prop :=
  p.resource a = p.resource b ∧ p.streamIdx a < p.streamIdx b

/-- The happens-before relation the protocol induces. The reflexive-transitive
    closure of counter-gated ∪ resource-ordered. -/
inductive happensBefore {tg : TaskGraph} (p : CounterProtocol tg) : Fin tg.n → Fin tg.n → Prop
  | counter  {a b : Fin tg.n} : counterGated  p a b → happensBefore p a b
  | resource {a b : Fin tg.n} : resourceOrdered p a b → happensBefore p a b
  | trans    {a b c : Fin tg.n} :
      happensBefore p a b → happensBefore p b c → happensBefore p a c

/-- A well-formed schedule: every counter's threshold equals its producer count,
    no task waits on a counter it itself increments, the data-dependency edges
    are a subset of the counter-or-resource graph, and there is a topological
    numbering of tasks under which both base relations point forward.

    The topological-order field (`scheduleOrder` + `*Forward`) is exactly what
    `list_schedule` already produces by construction: it assigns each task a
    start cycle s.t. counter waits and stream ordering only point to earlier
    cycles. We carry it as an explicit witness instead of re-deriving it from
    the scheduler simulator, which would require a state-machine encoding. -/
structure WellFormed {tg : TaskGraph} (p : CounterProtocol tg) where
  /-- Satisfiability: a counter is "all-complete" only when every producer
      increments it exactly once. -/
  satisfiable : ∀ c, p.threshold c = (List.range tg.n).countP
    (fun i => match (Nat.decLt i tg.n) with
      | isTrue h => c ∈ p.succs ⟨i, h⟩
      | isFalse _ => false)
  /-- No self-dependency: a task does not wait on its own succ counters. -/
  noSelfDep   : ∀ t c, c ∈ p.waits t → c ∈ p.succs t → False
  /-- Edge coverage: every data edge is enforced by the protocol. -/
  edgeCovered : ∀ (e : Fin tg.n × Fin tg.n), e ∈ tg.edges →
                  counterGated p e.1 e.2 ∨ resourceOrdered p e.1 e.2
  /-- Topological numbering supplied by the scheduler. -/
  scheduleOrder : Fin tg.n → Nat
  /-- Counter-gated edges point forward in `scheduleOrder`. -/
  cntForward : ∀ a b, counterGated p a b → scheduleOrder a < scheduleOrder b
  /-- Resource-ordered edges point forward in `scheduleOrder`. -/
  resForward : ∀ a b, resourceOrdered p a b → scheduleOrder a < scheduleOrder b

/-! ## Sub-lemmas (§2.T1) — proofs deferred. -/

/-- L1 (satisfiability): if `threshold c = |{i | c ∈ succs i}|`, every producer
    fires exactly once and the counter eventually saturates. -/
theorem satisfiability_lemma {tg : TaskGraph} (p : CounterProtocol tg)
    (wf : WellFormed p) (c : CounterId) :
    p.threshold c = (List.range tg.n).countP
      (fun i => match (Nat.decLt i tg.n) with
        | isTrue h => c ∈ p.succs ⟨i, h⟩
        | isFalse _ => false) :=
  wf.satisfiable c

/-- L2 (no-self-dep): a task never waits on a counter it itself increments. -/
theorem no_self_dep_lemma {tg : TaskGraph} (p : CounterProtocol tg)
    (wf : WellFormed p) (t : Fin tg.n) (c : CounterId)
    (hw : c ∈ p.waits t) (hs : c ∈ p.succs t) : False :=
  wf.noSelfDep t c hw hs

/-- L3 (edge coverage): every data edge has a counter or resource path. -/
theorem edge_covered_lemma {tg : TaskGraph} (p : CounterProtocol tg)
    (wf : WellFormed p) :
    ∀ e ∈ tg.edges, counterGated p e.1 e.2 ∨ resourceOrdered p e.1 e.2 :=
  wf.edgeCovered

/-! ## Main theorem (§2.T1). -/

/-- Main theorem: a well-formed protocol's happens-before relation covers
    every data-dependency edge.

    **Interface lemma.** The heavy lifting is the `WellFormed.edgeCovered`
    field, which this theorem merely lifts into `happensBefore`. Whether a
    *concrete* schedule satisfies `edgeCovered` is not proven here — it is
    checked executably, per compiled schedule, by `Plow/Verify.lean`
    (invoked from `plowc`'s lean-verify pipeline). This lemma pins down
    what that check buys: coverage of every data edge by the counter/
    resource happens-before relation. -/
theorem protocol_covers_deps {tg : TaskGraph} (p : CounterProtocol tg)
    (wf : WellFormed p) :
    ∀ e ∈ tg.edges, happensBefore p e.1 e.2 := by
  intro e he
  rcases wf.edgeCovered e he with hc | hr
  · exact happensBefore.counter hc
  · exact happensBefore.resource hr

/-- L4a (monotone in schedule order): every happens-before pair is strictly
    increasing in the scheduler-supplied topological numbering. Proven by
    structural induction on the `happensBefore` derivation. -/
theorem happensBefore_increases_order {tg : TaskGraph} (p : CounterProtocol tg)
    (wf : WellFormed p) :
    ∀ {a b : Fin tg.n}, happensBefore p a b →
      wf.scheduleOrder a < wf.scheduleOrder b := by
  intro a b h
  induction h with
  | counter hg  => exact wf.cntForward _ _ hg
  | resource hr => exact wf.resForward _ _ hr
  | trans _ _ ih1 ih2 => exact Nat.lt_trans ih1 ih2

/-- L4 (acyclicity): the combined counter-and-resource happens-before relation
    has no cycles. Immediate from `happensBefore_increases_order` —
    a self-loop would require `scheduleOrder t < scheduleOrder t`. -/
theorem happensBefore_acyclic {tg : TaskGraph} (p : CounterProtocol tg)
    (wf : WellFormed p) :
    ∀ t : Fin tg.n, ¬ happensBefore p t t := by
  intro t h
  exact Nat.lt_irrefl _ (happensBefore_increases_order p wf h)

end Plow.Protocol
