/-
# Plow.KvRing — endpoints `kv_ring.v1` and `vmm_trace.v1`.

`kv_ring.v1`: the sliding-window KV ring of a packed prefill launch. A launch writes every K/V
row of its spans before any attention read, so a span's writes must not land on a ring row its
own queries read (`dev_isa.h` SLIDING-WINDOW KV RING: `ring ≥ window + rows_written − 1`).
`launch_safe` derives that from the checked contract for absolute positions across any number
of wraps; `mask_eq_mod` justifies the kernel's `pos & (ring − 1)` addressing.

`vmm_trace.v1`: driver-level VMM traces (`memory::VmmOps`: reserve, create, map, set_access,
unmap, release, address_free) recorded from the runtime's real slot/prefix code under test
mocks. `step` is the transition relation; `traceOk` accepts a trace iff every step is enabled
(and, with `quiesce`, everything is returned). `step_preserves` proves the state invariant
(mappings reference live handles, are pairwise disjoint and lie inside reservations) is
inductive; `traceOk_sound` lifts it to every prefix of an accepted trace. Device completion
before reuse is not modelled: `VmmOps` carries no retirement events.
-/
import Lean.Data.Json

namespace Plow.KvRing

/-! ## Ring addressing -/

theorem mask_eq_mod (pos k : Nat) : pos &&& (2 ^ k - 1) = pos % 2 ^ k :=
  Nat.and_pow_two_sub_one_eq_mod pos k

/-- Positions inside one window of `ring` consecutive positions occupy distinct ring rows. -/
theorem mod_injective_window {ring lo p q : Nat} (hp : lo ≤ p) (hp' : p < lo + ring)
    (hq : lo ≤ q) (hq' : q < lo + ring) (h : p % ring = q % ring) : p = q := by
  have dp := Nat.div_add_mod p ring
  have dq := Nat.div_add_mod q ring
  rcases Nat.lt_trichotomy (p / ring) (q / ring) with lt | eq | gt
  · have := Nat.mul_le_mul_left ring (Nat.succ_le_of_lt lt)
    rw [Nat.mul_succ] at this
    omega
  · rw [eq] at dp
    omega
  · have := Nat.mul_le_mul_left ring (Nat.succ_le_of_lt gt)
    rw [Nat.mul_succ] at this
    omega

structure Span where
  slot : Nat
  /-- First position this launch writes for the request (`kvlen − qlen` of the stage). -/
  start : Nat
  /-- Rows written this launch (the stage's clipped `qlen`); 0 = idle entry. -/
  len : Nat
  deriving Repr, DecidableEq

structure Launch where
  ringLog : Nat
  window : Nat
  capacity : Nat
  spans : List Span
  deriving Repr

def Launch.ring (l : Launch) : Nat := 2 ^ l.ringLog

def Launch.Contract (l : Launch) : Prop :=
  0 < l.window ∧ l.ringLog < 32 ∧ l.capacity < 2 ^ 32 ∧
  (∀ s ∈ l.spans, s.len = 0 ∨ (l.window + s.len ≤ l.ring + 1 ∧ s.start + s.len ≤ l.capacity)) ∧
  ((l.spans.filter (·.len ≠ 0)).map (·.slot)).Nodup

instance (l : Launch) : Decidable l.Contract := by unfold Launch.Contract; exact inferInstance

/-- Query position `query` of a span reads key `key` iff `query − window < key ≤ query`. -/
def reads (window query key : Nat) : Prop := key ≤ query ∧ query < key + window

/-- No row a launch writes for a span aliases (mod ring) a different position the span's own
    queries read, whatever the absolute positions (any number of wraps). -/
theorem launch_safe (l : Launch) (h : l.Contract) (s : Span) (hs : s ∈ l.spans)
    {w q k : Nat} (hw : s.start ≤ w ∧ w < s.start + s.len) (hq : s.start ≤ q ∧ q < s.start + s.len)
    (hr : reads l.window q k) (hne : w ≠ k) : w % l.ring ≠ k % l.ring := by
  obtain ⟨_, _, _, hspans, _⟩ := h
  rcases hspans s hs with zero | ⟨fit, _⟩
  · omega
  intro heq
  apply hne
  unfold reads at hr
  -- Every position touched lies in [start + 1 − window, start + len) (truncated at 0).
  exact mod_injective_window (lo := s.start + 1 - l.window) (ring := l.ring)
    (by omega) (by omega) (by omega) (by omega) heq

/-- The kernel's masked row equals the modular row, so `launch_safe` holds for `pos & (ring−1)`. -/
theorem launch_safe_masked (l : Launch) (h : l.Contract) (s : Span) (hs : s ∈ l.spans)
    {w q k : Nat} (hw : s.start ≤ w ∧ w < s.start + s.len) (hq : s.start ≤ q ∧ q < s.start + s.len)
    (hr : reads l.window q k) (hne : w ≠ k) : w &&& (l.ring - 1) ≠ k &&& (l.ring - 1) := by
  unfold Launch.ring
  rw [mask_eq_mod, mask_eq_mod]
  exact launch_safe l h s hs hw hq hr hne

theorem inj_of_nodup_map {α β : Type} {f : α → β} : ∀ {l : List α}, (l.map f).Nodup →
    ∀ {a b : α}, a ∈ l → b ∈ l → f a = f b → a = b
  | [], _, _, _, ha, _, _ => absurd ha (List.not_mem_nil _)
  | x :: xs, nd, a, b, ha, hb, hf => by
    rw [List.map_cons, List.nodup_cons] at nd
    obtain ⟨notin, nd⟩ := nd
    rcases List.mem_cons.mp ha with ea | ta <;> rcases List.mem_cons.mp hb with eb | tb
    · rw [ea, eb]
    · exact absurd (ea ▸ hf ▸ List.mem_map_of_mem f tb) notin
    · exact absurd (eb ▸ hf.symm ▸ List.mem_map_of_mem f ta) notin
    · exact inj_of_nodup_map nd ta tb hf

/-- Non-idle spans of one launch that share a slot are the same span: one ring per request. -/
theorem slots_distinct (l : Launch) (h : l.Contract) (a b : Span) (ha : a ∈ l.spans) (hb : b ∈ l.spans)
    (hal : a.len ≠ 0) (hbl : b.len ≠ 0) (hslot : a.slot = b.slot) : a = b := by
  obtain ⟨_, _, _, _, nodup⟩ := h
  have ma : a ∈ l.spans.filter (·.len ≠ 0) := List.mem_filter.mpr ⟨ha, by simpa using hal⟩
  have mb : b ∈ l.spans.filter (·.len ≠ 0) := List.mem_filter.mpr ⟨hb, by simpa using hbl⟩
  exact inj_of_nodup_map nodup ma mb hslot

def checkLaunch (l : Launch) : Bool := decide l.Contract

theorem checkLaunch_sound (l : Launch) (h : checkLaunch l = true) : l.Contract := of_decide_eq_true h

/-! ## VMM lifecycle traces -/

inductive Event
  | reserve (va bytes : Nat)
  | addressFree (va bytes : Nat)
  | create (handle bytes : Nat)
  | release (handle : Nat)
  | map (va bytes handle : Nat)
  | unmap (va bytes : Nat)
  | access (va bytes : Nat)
  deriving Repr, DecidableEq

structure Mapping where
  va : Nat
  bytes : Nat
  handle : Nat
  deriving Repr, DecidableEq

structure State where
  reservations : List (Nat × Nat)
  handles : List Nat
  mappings : List Mapping
  deriving Repr

def State.empty : State := ⟨[], [], []⟩

def overlaps (a b x y : Nat) : Bool := decide (a < x + y) && decide (x < a + b)

def inside (va bytes : Nat) (r : Nat × Nat) : Bool := decide (r.1 ≤ va) && decide (va + bytes ≤ r.1 + r.2)

/-- Bytes of `[va, va+bytes)` covered by mappings (each fully inside it). With pairwise-disjoint
    mappings, `covered = bytes` means the range is fully mapped. -/
def covered (ms : List Mapping) (va bytes : Nat) : Nat :=
  (ms.filter fun m => inside m.va m.bytes (va, bytes)).foldr (fun m acc => m.bytes + acc) 0

def step (s : State) : Event → Option State
  | .reserve va bytes =>
    if 0 < bytes ∧ s.reservations.all (fun r => !overlaps va bytes r.1 r.2) then
      some { s with reservations := (va, bytes) :: s.reservations } else none
  | .addressFree va bytes =>
    if (va, bytes) ∈ s.reservations ∧ s.mappings.all (fun m => !overlaps va bytes m.va m.bytes) then
      some { s with reservations := s.reservations.erase (va, bytes) } else none
  | .create handle _ =>
    if handle ∉ s.handles then some { s with handles := handle :: s.handles } else none
  | .release handle =>
    if handle ∈ s.handles ∧ s.mappings.all (fun m => m.handle != handle) then
      some { s with handles := s.handles.erase handle } else none
  | .map va bytes handle =>
    if 0 < bytes ∧ handle ∈ s.handles ∧ s.reservations.any (inside va bytes) ∧
        s.mappings.all (fun m => !overlaps va bytes m.va m.bytes) then
      some { s with mappings := ⟨va, bytes, handle⟩ :: s.mappings } else none
  | .unmap va bytes =>
    match s.mappings.find? (fun m => m.va == va && m.bytes == bytes) with
    | some m => some { s with mappings := s.mappings.erase m }
    | none => none
  | .access va bytes =>
    if 0 < bytes ∧ covered s.mappings va bytes = bytes then some s else none

/-- Mappings reference live handles, are pairwise disjoint, and lie inside reservations. -/
def State.Inv (s : State) : Prop :=
  (∀ m ∈ s.mappings, m.handle ∈ s.handles) ∧
  s.mappings.Pairwise (fun a b => overlaps a.va a.bytes b.va b.bytes = false) ∧
  (∀ m ∈ s.mappings, ∃ r ∈ s.reservations, inside m.va m.bytes r = true)

theorem empty_inv : State.empty.Inv := by
  simp [State.Inv, State.empty]

theorem overlaps_comm (a b x y : Nat) : overlaps a b x y = overlaps x y a b := by
  unfold overlaps; rw [Bool.and_comm]

/-- Free a reservation only when no mapping intersects it; every mapping still has a reservation. -/
theorem inside_of_disjoint {va bytes : Nat} {m : Mapping} {r : Nat × Nat}
    (hin : inside m.va m.bytes r = true) (hpos : 0 < m.bytes) (hr : r = (va, bytes))
    (hdis : overlaps va bytes m.va m.bytes = false) : False := by
  subst hr
  simp only [inside, overlaps, Bool.and_eq_true, decide_eq_true_eq, Bool.and_eq_false_iff,
    decide_eq_false_iff_not] at hin hdis
  omega

/-- `step_preserves` needs positive mapping sizes, which `map` enforces; carry it in the state
    invariant of the trace checker. -/
def State.Pos (s : State) : Prop := ∀ m ∈ s.mappings, 0 < m.bytes

theorem step_preserves {s s' : State} {e : Event} (hi : s.Inv) (hp : s.Pos) (h : step s e = some s') :
    s'.Inv ∧ s'.Pos := by
  obtain ⟨hlive, hdisj, hres⟩ := hi
  cases e with
  | reserve va bytes =>
    simp only [step] at h
    split at h
    · simp only [Option.some.injEq] at h; subst h
      refine ⟨⟨hlive, hdisj, fun m hm => ?_⟩, hp⟩
      obtain ⟨r, hr, hin⟩ := hres m hm
      exact ⟨r, List.mem_cons_of_mem _ hr, hin⟩
    · simp at h
  | addressFree va bytes =>
    simp only [step] at h
    split at h
    · rename_i hc
      simp only [Option.some.injEq] at h; subst h
      refine ⟨⟨hlive, hdisj, fun m hm => ?_⟩, hp⟩
      obtain ⟨r, hr, hin⟩ := hres m hm
      refine ⟨r, ?_, hin⟩
      by_cases hrv : r = (va, bytes)
      · have hdis := List.all_eq_true.mp hc.2 m hm
        simp only [Bool.not_eq_true'] at hdis
        exact (inside_of_disjoint hin (hp m hm) hrv hdis).elim
      · exact (List.mem_erase_of_ne hrv).mpr hr
    · simp at h
  | create handle _ =>
    simp only [step] at h
    split at h
    · simp only [Option.some.injEq] at h; subst h
      exact ⟨⟨fun m hm => List.mem_cons_of_mem _ (hlive m hm), hdisj, hres⟩, hp⟩
    · simp at h
  | release handle =>
    simp only [step] at h
    split at h
    · rename_i hc
      simp only [Option.some.injEq] at h; subst h
      refine ⟨⟨fun m hm => ?_, hdisj, hres⟩, hp⟩
      have hne := List.all_eq_true.mp hc.2 m hm
      simp only [bne_iff_ne, ne_eq] at hne
      exact (List.mem_erase_of_ne hne).mpr (hlive m hm)
    · simp at h
  | map va bytes handle =>
    simp only [step] at h
    split at h
    · rename_i hc
      obtain ⟨hpos, hh, hin, hfree⟩ := hc
      simp only [Option.some.injEq] at h; subst h
      refine ⟨⟨fun m hm => ?_, ?_, fun m hm => ?_⟩, fun m hm => ?_⟩
      · rcases List.mem_cons.mp hm with rfl | hm
        · exact hh
        · exact hlive m hm
      · refine List.Pairwise.cons (fun m hm => ?_) hdisj
        have := List.all_eq_true.mp hfree m hm
        simpa using this
      · rcases List.mem_cons.mp hm with rfl | hm
        · obtain ⟨r, hr, hi⟩ := List.any_eq_true.mp hin
          exact ⟨r, hr, hi⟩
        · exact hres m hm
      · rcases List.mem_cons.mp hm with rfl | hm
        · exact hpos
        · exact hp m hm
    · simp at h
  | unmap va bytes =>
    simp only [step] at h
    split at h
    · rename_i m _
      simp only [Option.some.injEq] at h; subst h
      have sub : ∀ x, x ∈ s.mappings.erase m → x ∈ s.mappings := fun x hx => List.mem_of_mem_erase hx
      exact ⟨⟨fun x hx => hlive x (sub x hx), List.Pairwise.sublist (List.erase_sublist _ _) hdisj,
        fun x hx => hres x (sub x hx)⟩, fun x hx => hp x (sub x hx)⟩
    · simp at h
  | access va bytes =>
    simp only [step] at h
    split at h
    · simp only [Option.some.injEq] at h; subst h
      exact ⟨⟨hlive, hdisj, hres⟩, hp⟩
    · simp at h

/-- Run the trace; `none` = some step was not enabled. -/
def run : State → List Event → Option State
  | s, [] => some s
  | s, e :: es => (step s e).bind fun s' => run s' es

def State.quiescent (s : State) : Bool := s.reservations.isEmpty && s.handles.isEmpty && s.mappings.isEmpty

def traceOk (quiesce : Bool) (events : List Event) : Bool :=
  match run State.empty events with
  | some s => !quiesce || s.quiescent
  | none => false

theorem run_preserves : ∀ (es : List Event) {s s' : State}, s.Inv → s.Pos → run s es = some s' → s'.Inv
  | [], _, _, hi, _, h => by simp only [run, Option.some.injEq] at h; exact h ▸ hi
  | e :: es, s, s', hi, hp, h => by
    simp only [run] at h
    cases hs : step s e with
    | none => simp [hs] at h
    | some t =>
      simp only [hs, Option.bind_some] at h
      obtain ⟨ti, tp⟩ := step_preserves hi hp hs
      exact run_preserves es ti tp h

/-- Every prefix of an accepted trace leaves a state satisfying the invariant: no mapping
    outlives its handle or reservation, and no two mappings overlap. With `quiesce`, every
    reservation, handle and mapping has been returned exactly once by the end. -/
theorem traceOk_sound (quiesce : Bool) (events : List Event) (h : traceOk quiesce events = true) :
    ∀ n, ∃ s, run State.empty (events.take n) = some s ∧ s.Inv := by
  intro n
  have hfull : ∃ s, run State.empty events = some s := by
    unfold traceOk at h
    split at h
    · rename_i s hs; exact ⟨s, hs⟩
    · simp at h
  have prefix_ok : ∀ (es : List Event) (s : State) (k : Nat), (run s es).isSome →
      (run s (es.take k)).isSome := by
    intro es
    induction es with
    | nil => intro s k _; simp [run]
    | cons e rest ih =>
      intro s k hsome
      cases k with
      | zero => simp [run]
      | succ k =>
        simp only [run, List.take_succ_cons] at hsome ⊢
        cases hs : step s e with
        | none => simp [hs] at hsome
        | some t => simp only [hs, Option.bind_some] at hsome ⊢; exact ih t k hsome
  obtain ⟨s, hs⟩ := hfull
  have := prefix_ok events State.empty n (by simp [hs])
  obtain ⟨t, ht⟩ := Option.isSome_iff_exists.mp this
  exact ⟨t, ht, run_preserves _ empty_inv (by simp [State.Pos, State.empty]) ht⟩

/-! ## JSON -/

open Lean (Json)

def exactKeys (j : Json) (keys : List String) : Except String Unit := do
  let obj ← j.getObj?
  let present := obj.toArray.map (·.1) |>.toList
  for k in present do
    if !keys.contains k then throw s!"unknown field {k}"
  for k in keys do
    if !present.contains k then throw s!"missing field {k}"

def nat (j : Json) (k : String) : Except String Nat := j.getObjValAs? Nat k

def parseSpan (j : Json) : Except String Span := do
  exactKeys j ["slot", "start", "len"]
  return { slot := (← nat j "slot"), start := (← nat j "start"), len := (← nat j "len") }

def parseLaunch (j : Json) : Except String Launch := do
  exactKeys j ["ring_log", "window", "capacity", "spans"]
  let spans ← j.getObjValAs? (List Json) "spans"
  return {
    ringLog := (← nat j "ring_log"), window := (← nat j "window"),
    capacity := (← nat j "capacity"), spans := (← spans.mapM parseSpan) }

def runRing (j : Json) : Except String String := do
  exactKeys j ["schema", "launches"]
  if (← nat j "schema") != 1 then throw "unsupported kv_ring schema"
  let raw ← j.getObjValAs? (List Json) "launches"
  if raw.isEmpty then throw "no launches"
  let launches ← raw.mapM parseLaunch
  for (l, i) in launches.zip (List.range launches.length) do
    if !checkLaunch l then throw s!"launch {i}: ring/window/slot contract violated"
  return s!"checkLaunch_sound + launch_safe_masked: {launches.length} launches write no ring row their own queries read; slots distinct; kernel arithmetic outside scope"

def parseEvent (j : Json) : Except String Event := do
  match ← j.getObjValAs? String "op" with
  | "reserve" => exactKeys j ["op", "va", "bytes"]; return .reserve (← nat j "va") (← nat j "bytes")
  | "address_free" => exactKeys j ["op", "va", "bytes"]; return .addressFree (← nat j "va") (← nat j "bytes")
  | "create" => exactKeys j ["op", "handle", "bytes"]; return .create (← nat j "handle") (← nat j "bytes")
  | "release" => exactKeys j ["op", "handle"]; return .release (← nat j "handle")
  | "map" => exactKeys j ["op", "va", "bytes", "handle"]; return .map (← nat j "va") (← nat j "bytes") (← nat j "handle")
  | "unmap" => exactKeys j ["op", "va", "bytes"]; return .unmap (← nat j "va") (← nat j "bytes")
  | "access" => exactKeys j ["op", "va", "bytes"]; return .access (← nat j "va") (← nat j "bytes")
  | other => throw s!"unknown VMM op {other}"

def runTrace (j : Json) : Except String String := do
  exactKeys j ["schema", "quiesce", "events"]
  if (← nat j "schema") != 1 then throw "unsupported vmm_trace schema"
  let quiesce ← j.getObjValAs? Bool "quiesce"
  let raw ← j.getObjValAs? (List Json) "events"
  let events ← raw.mapM parseEvent
  -- Locate the first disabled step for the diagnostic; acceptance is `traceOk` alone.
  let mut s := State.empty
  for (e, i) in events.zip (List.range events.length) do
    match step s e with
    | some t => s := t
    | none => throw s!"event {i} ({repr e}) is not enabled"
  if !traceOk quiesce events then throw "trace ends with unreturned reservations, handles or mappings"
  return s!"traceOk_sound: {events.length} VMM events keep mappings on live handles inside reservations, pairwise disjoint{if quiesce then "; everything returned" else ""}; device completion before reuse is outside this model"

end Plow.KvRing
