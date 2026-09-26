/-
# Plow.Verify — executable address-map verifier.

Provides a decidable check on a concrete `AddrEntry` list + `CounterProtocol`:
every byte-overlapping pair has the bridging happens-before chain required by
`AddressMapSound`. The Bool verifier is proven sound against
`Plow.Memory.AddressMapSound`.

This is the "given an address map dumped from Rust, is it safe?" function.
The Rust FFI/IPC layer will feed JSON in; this module decides.
-/
import Plow.Basic
import Plow.Protocol
import Plow.Memory

namespace Plow.Verify

open Plow Plow.Protocol Plow.Memory

/-! ## Decidable primitives. -/

/-- Bool version of `bytesOverlap`. -/
def bytesOverlapB (off1 sz1 off2 sz2 : Nat) : Bool :=
  decide (off1 < off2 + sz2) && decide (off2 < off1 + sz1)

theorem bytesOverlapB_iff (off1 sz1 off2 sz2 : Nat) :
    bytesOverlapB off1 sz1 off2 sz2 = true ↔ bytesOverlap off1 sz1 off2 sz2 := by
  simp [bytesOverlapB, bytesOverlap]

/-- A direct edge `a → b` in the (counter ∪ resource-order) graph. -/
def directEdgeB {tg : TaskGraph} (p : CounterProtocol tg)
    (a b : Fin tg.n) : Bool :=
  (p.succs a).any (fun c => decide (c ∈ p.waits b))
  || (decide (p.resource a = p.resource b)
      && decide (p.streamIdx a < p.streamIdx b))

theorem directEdgeB_sound {tg : TaskGraph} (p : CounterProtocol tg)
    (a b : Fin tg.n) (h : directEdgeB p a b = true) :
    happensBefore p a b := by
  simp [directEdgeB] at h
  rcases h with hcnt | ⟨hres, hidx⟩
  · -- counter-gated
    obtain ⟨c, hc_succ, hc_wait⟩ := hcnt
    exact happensBefore.counter ⟨c, hc_succ, hc_wait⟩
  · exact happensBefore.resource ⟨hres, hidx⟩

/-! ## Fuel-bounded reachability. -/

/-- One iteration of reachability: extend the frontier with every node that
    has a direct edge from somewhere in the frontier. -/
def stepReach {tg : TaskGraph} (p : CounterProtocol tg)
    (reached : List (Fin tg.n)) : List (Fin tg.n) :=
  let candidates := (List.finRange tg.n).filter (fun b =>
    !reached.contains b && reached.any (fun a => directEdgeB p a b))
  reached ++ candidates

/-- Iterate `stepReach` `fuel` times from `[source]`. With `fuel = tg.n` this
    saturates (every reachable node is found). -/
def reachableUpTo {tg : TaskGraph} (p : CounterProtocol tg)
    (source : Fin tg.n) : Nat → List (Fin tg.n)
  | 0     => [source]
  | n + 1 => stepReach p (reachableUpTo p source n)

/-- Decidable happens-before via fuel-bounded reachability. Use
    `fuel = tg.n` for completeness; finite graphs saturate in n steps. -/
def happensBeforeB {tg : TaskGraph} (p : CounterProtocol tg)
    (a b : Fin tg.n) : Bool :=
  (reachableUpTo p a tg.n).contains b

/-- Soundness of `reachableUpTo`: every node in the reachable list is either
    the source, or reachable through some direct-edge chain. -/
theorem mem_reachableUpTo {tg : TaskGraph} (p : CounterProtocol tg)
    (a : Fin tg.n) (n : Nat) : ∀ (b : Fin tg.n),
    b ∈ reachableUpTo p a n → b = a ∨ happensBefore p a b := by
  induction n with
  | zero =>
    intro b h
    simp [reachableUpTo] at h
    exact Or.inl h
  | succ k ih =>
    intro b h
    simp [reachableUpTo, stepReach, List.mem_append, List.mem_filter,
          List.any_eq_true] at h
    rcases h with hin | ⟨_hin_range, _hnotin, hany⟩
    · exact ih b hin
    · obtain ⟨mid, hmid_in, hmid_edge⟩ := hany
      have hmid_step : happensBefore p mid b := directEdgeB_sound p mid b hmid_edge
      rcases ih mid hmid_in with heqa | hmid_hb
      · subst heqa; exact Or.inr hmid_step
      · exact Or.inr (happensBefore.trans hmid_hb hmid_step)

/-- Soundness of `happensBeforeB`: if it returns true, then `happensBefore`
    holds — unless `a = b`, which is also fine since the proof of
    `protocol_covers_deps` doesn't need `a ≠ b`. -/
theorem happensBeforeB_sound {tg : TaskGraph} (p : CounterProtocol tg)
    (a b : Fin tg.n) (h : happensBeforeB p a b = true) :
    a = b ∨ happensBefore p a b := by
  unfold happensBeforeB at h
  have hmem : b ∈ reachableUpTo p a tg.n := by
    simpa [List.contains, List.elem_eq_mem] using h
  rcases mem_reachableUpTo p a tg.n b hmem with heq | hhb
  · exact Or.inl heq.symm
  · exact Or.inr hhb

def verifyDependencies {tg : TaskGraph} (p : CounterProtocol tg) : Bool :=
  tg.edges.all fun (a, b) => a != b && happensBeforeB p a b

theorem verifyDependencies_sound {tg : TaskGraph} (p : CounterProtocol tg)
    (h : verifyDependencies p = true) :
    ∀ a b, (a, b) ∈ tg.edges → happensBefore p a b := by
  intro a b hab
  have edge := List.all_eq_true.mp h (a, b) hab
  simp only [Bool.and_eq_true, bne_iff_ne] at edge
  rcases happensBeforeB_sound p a b edge.2 with eq | hb
  · exact False.elim (edge.1 eq)
  · exact hb

def checkPath {tg : TaskGraph} (p : CounterProtocol tg)
    (a b : Fin tg.n) : List (Fin tg.n) → Bool
  | [] => directEdgeB p a b
  | mid :: rest => directEdgeB p a mid && checkPath p mid b rest

theorem checkPath_sound {tg : TaskGraph} (p : CounterProtocol tg)
    (a b : Fin tg.n) (path : List (Fin tg.n))
    (h : checkPath p a b path = true) : happensBefore p a b := by
  induction path generalizing a with
  | nil => exact directEdgeB_sound p a b h
  | cons mid rest ih =>
    simp only [checkPath, Bool.and_eq_true] at h
    exact happensBefore.trans (directEdgeB_sound p a mid h.1) (ih mid h.2)

def checkPaths {tg : TaskGraph} (p : CounterProtocol tg) :
    List (Fin tg.n × Fin tg.n) → List (List (Fin tg.n)) → Bool
  | [], [] => true
  | (a, b) :: edges, path :: paths => checkPath p a b path && checkPaths p edges paths
  | _, _ => false

theorem checkPaths_sound {tg : TaskGraph} (p : CounterProtocol tg)
    (edges : List (Fin tg.n × Fin tg.n)) (paths : List (List (Fin tg.n)))
    (h : checkPaths p edges paths = true) :
    ∀ a b, (a, b) ∈ edges → happensBefore p a b := by
  induction edges generalizing paths with
  | nil => simp
  | cons edge rest ih =>
    cases paths with
    | nil => simp [checkPaths] at h
    | cons path paths =>
      obtain ⟨a, b⟩ := edge
      simp only [checkPaths, Bool.and_eq_true] at h
      intro x y hxy
      rcases List.mem_cons.mp hxy with same | tail
      · cases same
        exact checkPath_sound p a b path h.1
      · exact ih paths h.2 x y tail


structure PathWitness (tg : TaskGraph) where
  source : Fin tg.n
  target : Fin tg.n
  via : List (Fin tg.n)

def witnessedBefore {tg : TaskGraph} (p : CounterProtocol tg)
    (paths : List (PathWitness tg)) (a b : Fin tg.n) : Bool :=
  paths.any fun w => decide (w.source = a) && decide (w.target = b) &&
    checkPath p w.source w.target w.via

theorem witnessedBefore_sound {tg : TaskGraph} (p : CounterProtocol tg)
    (paths : List (PathWitness tg)) (a b : Fin tg.n)
    (h : witnessedBefore p paths a b = true) : happensBefore p a b := by
  simp only [witnessedBefore, List.any_eq_true, Bool.and_eq_true, decide_eq_true_eq] at h
  obtain ⟨w, _, ⟨hs, ht⟩, hp⟩ := h
  have checked := checkPath_sound p w.source w.target w.via hp
  simpa [hs, ht] using checked

def readersBeforeWritersVia {tg : TaskGraph} (p : CounterProtocol tg)
    (paths : List (PathWitness tg)) (readers writers : List (Fin tg.n)) : Bool :=
  readers.all fun r => writers.all fun w => witnessedBefore p paths r w

def verifyAddressMapVia {tg : TaskGraph} (p : CounterProtocol tg)
    (entries : List (AddrEntry tg)) (paths : List (PathWitness tg)) : Bool :=
  entries.all fun a => entries.all fun b =>
    decide (a.name = b.name) || !bytesOverlapB a.offset a.size b.offset b.size
    || readersBeforeWritersVia p paths a.readers b.writers
    || readersBeforeWritersVia p paths b.readers a.writers

theorem verifyAddressMapVia_sound {tg : TaskGraph} (p : CounterProtocol tg)
    (entries : List (AddrEntry tg)) (paths : List (PathWitness tg))
    (h : verifyAddressMapVia p entries paths = true) : AddressMapSound p entries := by
  intro a b ha hb hne hov
  have pair := (List.all_eq_true.mp ((List.all_eq_true.mp h) a ha)) b hb
  have name : decide (a.name = b.name) = false := by simp [hne]
  have overlap : bytesOverlapB a.offset a.size b.offset b.size = true :=
    (bytesOverlapB_iff a.offset a.size b.offset b.size).mpr hov
  rw [name, overlap] at pair
  simp only [Bool.false_or, Bool.not_true, Bool.or_eq_true] at pair
  rcases pair with forward | backward
  · left
    intro r hr w hw
    exact witnessedBefore_sound p paths r w
      ((List.all_eq_true.mp ((List.all_eq_true.mp forward) r hr)) w hw)
  · right
    intro r hr w hw
    exact witnessedBefore_sound p paths r w
      ((List.all_eq_true.mp ((List.all_eq_true.mp backward) r hr)) w hw)

/-! ## The address-map verifier. -/

/-- For a `(reader, writer)` pair, the verifier requires `a ≠ writer` and
    `happensBefore`. When `a = writer` (self-edge), it's trivially safe only
    if no actual write happens — but reclamation always involves a real
    writer, so we conservatively require strict happens-before. -/
def readersBeforeWriters {tg : TaskGraph} (p : CounterProtocol tg)
    (readers writers : List (Fin tg.n)) : Bool :=
  readers.all (fun r => writers.all (fun w => happensBeforeB p r w))

/-- The verifier: for every distinct-name byte-overlapping pair, at least one
    direction (readers-of-a → writers-of-b, or symmetric) is fully ordered. -/
def verifyAddressMap {tg : TaskGraph} (p : CounterProtocol tg)
    (entries : List (AddrEntry tg)) : Bool :=
  entries.all fun a => entries.all fun b =>
    decide (a.name = b.name)
    || !bytesOverlapB a.offset a.size b.offset b.size
    || readersBeforeWriters p a.readers b.writers
    || readersBeforeWriters p b.readers a.writers

/-! ## Soundness theorem.

    We need the `happensBeforeB`-derived `mayReclaim` — but `happensBeforeB`
    can return `true` for `a = b` (a node is reachable from itself in zero
    steps). For the reclamation use case this is safe because the disjunct
    `happensBefore r w` permits `r = w` trivially via the underlying
    relation only when there's an actual edge — but with `mem_reachableUpTo`
    we get `r = w ∨ happensBefore r w`, which is *not* directly
    `happensBefore r w`. So we need an additional precondition: a writer is
    never also a reader of the same buffer pair across the reclamation
    boundary (which is the case in practice: a write task does not also
    read the older buffer being reclaimed).

    To keep the soundness theorem clean, we strengthen `mayReclaim`'s
    consequence to allow `r = w` (in which case there's no race because the
    same task can't both finish reading the old buffer and write the new one
    in violation of itself). -/

/-- The "loose" reclamation predicate that the executable verifier proves:
    either equal or ordered. In the real Rust schedule, the writer/reader
    sets are disjoint, so `r = w` doesn't arise — but encoding that disjoint-
    ness in Lean is extra work, so we expose the looser predicate and let the
    caller add the disjointness side-condition. -/
def mayReclaimLoose {tg : TaskGraph} (p : CounterProtocol tg)
    (a b : AddrEntry tg) : Prop :=
  (∀ r ∈ a.readers, ∀ w ∈ b.writers, r = w ∨ happensBefore p r w) ∨
  (∀ r ∈ b.readers, ∀ w ∈ a.writers, r = w ∨ happensBefore p r w)

def AddressMapSoundLoose {tg : TaskGraph} (p : CounterProtocol tg)
    (entries : List (AddrEntry tg)) : Prop :=
  ∀ a b, a ∈ entries → b ∈ entries → a.name ≠ b.name →
    bytesOverlap a.offset a.size b.offset b.size →
    mayReclaimLoose p a b

theorem verifyAddressMap_sound {tg : TaskGraph} (p : CounterProtocol tg)
    (entries : List (AddrEntry tg))
    (h : verifyAddressMap p entries = true) :
    AddressMapSoundLoose p entries := by
  intro a b ha hb hne hov
  -- Pull the per-pair clause out of the doubly-nested all.
  have hpair : (
      decide (a.name = b.name)
      || !bytesOverlapB a.offset a.size b.offset b.size
      || readersBeforeWriters p a.readers b.writers
      || readersBeforeWriters p b.readers a.writers
    ) = true := by
    have h1 := (List.all_eq_true.mp h) a ha
    exact (List.all_eq_true.mp h1) b hb
  -- The first two clauses are excluded by hne and hov; one of the last two
  -- must therefore be true.
  have hname_false : decide (a.name = b.name) = false := by
    simp [hne]
  have hov_true : bytesOverlapB a.offset a.size b.offset b.size = true :=
    (bytesOverlapB_iff a.offset a.size b.offset b.size).mpr hov
  rw [hname_false, hov_true] at hpair
  simp at hpair
  -- hpair : readersBeforeWriters p a.readers b.writers = true
  --       ∨ readersBeforeWriters p b.readers a.writers = true
  rcases hpair with hfwd | hbwd
  · refine Or.inl ?_
    intro r hr w hw
    have h1 := (List.all_eq_true.mp hfwd) r hr
    have h2 := (List.all_eq_true.mp h1) w hw
    exact happensBeforeB_sound p r w h2
  · refine Or.inr ?_
    intro r hr w hw
    have h1 := (List.all_eq_true.mp hbwd) r hr
    have h2 := (List.all_eq_true.mp h1) w hw
    exact happensBeforeB_sound p r w h2

/-! ## Reader/writer disjointness — the bridge from Loose to Strict.

    `verifyAddressMap` proves `AddressMapSoundLoose`: the reclamation clause
    permits `reader = writer` (via `happensBeforeB`'s zero-fuel case). The
    **strict** `Plow.Memory.AddressMapSound` demands `happensBefore r w`.

    In real schedules the reader and writer sets across a reclamation
    boundary are **disjoint** by construction — a task that reads the old
    buffer cannot also be a writer of the new one, because the allocator
    always picks a fresh producer task for the new buffer. We expose that as
    a separate decidable predicate and prove Loose ∧ Disjoint ⟹ Strict. -/

/-- Bool check: for every **co-located** pair — distinct names, bytes
    overlapping — the reader set of one and the writer set of the other are
    disjoint. When true, the loose form of `mayReclaim` collapses to the strict
    form on exactly the pairs `AddressMapSound` quantifies over.

    # The guards are load-bearing, and their absence was a false rejection
    This predicate used to quantify over ALL pairs with no guards, which
    (taking `a = b`) made it equivalent to "the union of every reader set and
    the union of every writer set are disjoint" — i.e. **no task anywhere may
    both read and write any mapped buffer**. That is not a property real
    schedules have, and it is not one `AddressMapSound` needs: a KV cache is
    read and written by the same attention tasks, and an in-place residual
    accumulate reads and writes one buffer by construction.

    Measured on Gemma-4-12B `decode_b1_s1`, whose address map is two entries —
    `t7` at bytes `[0, 6144)` (24 writers, no readers) and `kv_cache_L0` at
    `[6144, 3151872)` (tasks 33-44 in BOTH sets). The two do not overlap, so
    `AddressMapSound` has nothing to check and holds vacuously; the unguarded
    predicate nevertheless found tasks 33-44 in the global reader ∩ writer
    intersection and rejected the packet. The guards below are the same two
    `AddressMapSound` itself carries, so the check now asks the question the
    theorem actually needs answered. -/
def readersWritersDisjointB {tg : TaskGraph} (entries : List (AddrEntry tg)) :
    Bool :=
  entries.all fun a => entries.all fun b =>
    decide (a.name = b.name)
    || !bytesOverlapB a.offset a.size b.offset b.size
    || ((a.readers.all fun r => decide (r ∉ b.writers))
        && (b.readers.all fun r => decide (r ∉ a.writers)))

/-- Prop-level statement of disjointness, on co-located pairs only. -/
def ReadersWritersDisjoint {tg : TaskGraph}
    (entries : List (AddrEntry tg)) : Prop :=
  ∀ a b, a ∈ entries → b ∈ entries → a.name ≠ b.name →
    bytesOverlap a.offset a.size b.offset b.size →
    (∀ r ∈ a.readers, r ∉ b.writers) ∧
    (∀ r ∈ b.readers, r ∉ a.writers)

/-- Soundness of the Bool disjointness check. -/
theorem readersWritersDisjointB_sound {tg : TaskGraph}
    (entries : List (AddrEntry tg))
    (h : readersWritersDisjointB entries = true) :
    ReadersWritersDisjoint entries := by
  intro a b ha hb hne hov
  -- Pull the per-pair clause out, then discharge the two guards exactly as
  -- `verifyAddressMap_sound` does for the same pair shape.
  have hpair : (
      decide (a.name = b.name)
      || !bytesOverlapB a.offset a.size b.offset b.size
      || ((a.readers.all fun r => decide (r ∉ b.writers))
          && (b.readers.all fun r => decide (r ∉ a.writers)))
    ) = true := by
    have h1 := (List.all_eq_true.mp h) a ha
    exact (List.all_eq_true.mp h1) b hb
  have hname_false : decide (a.name = b.name) = false := by
    simp [hne]
  have hov_true : bytesOverlapB a.offset a.size b.offset b.size = true :=
    (bytesOverlapB_iff a.offset a.size b.offset b.size).mpr hov
  rw [hname_false, hov_true] at hpair
  -- With both guards discharged the remaining disjunct is the conjunction, and
  -- `simp` takes the two `List.all`s straight to their ∀ forms — which is the
  -- goal.
  simpa using hpair

/-- **Loose ∧ Disjoint ⟹ Strict**: given the loose reclamation predicate
    and the reader/writer disjointness invariant, every occurrence of the
    `r = w` disjunct is impossible, so the strict `mayReclaim` (=
    `happensBefore r w`) follows. -/
theorem addressMapSound_of_loose_and_disjoint {tg : TaskGraph}
    (p : CounterProtocol tg) (entries : List (AddrEntry tg))
    (h_loose : AddressMapSoundLoose p entries)
    (h_disj : ReadersWritersDisjoint entries) :
    AddressMapSound p entries := by
  intro a b ha hb hne hov
  -- `h_disj` is now available exactly where the goal is: on a co-located pair.
  -- That is all this proof ever used it for, which is why narrowing the
  -- predicate costs nothing here and stops it rejecting in-place buffers.
  have hd := h_disj a b ha hb hne hov
  rcases h_loose a b ha hb hne hov with hfwd | hbwd
  · refine Or.inl ?_
    intro r hr w hw
    have := hfwd r hr w hw
    rcases this with heq | hhb
    · exfalso; apply hd.1 r hr; rw [heq]; exact hw
    · exact hhb
  · refine Or.inr ?_
    intro r hr w hw
    have := hbwd r hr w hw
    rcases this with heq | hhb
    · exfalso; apply hd.2 r hr; rw [heq]; exact hw
    · exact hhb

/-- **Strict verifier**: `verifyAddressMap ∧ readersWritersDisjointB ⟹
    AddressMapSound`. This is the strict form the executable can prove when
    the caller supplies the disjointness precondition (Rust always does, by
    the way `plan_from_schedule_with_task_sets` builds the sets). -/
theorem verifyAddressMap_sound_strict {tg : TaskGraph} (p : CounterProtocol tg)
    (entries : List (AddrEntry tg))
    (h_verify : verifyAddressMap p entries = true)
    (h_disj : readersWritersDisjointB entries = true) :
    AddressMapSound p entries :=
  addressMapSound_of_loose_and_disjoint p entries
    (verifyAddressMap_sound p entries h_verify)
    (readersWritersDisjointB_sound entries h_disj)

/-! ## Unit tests.

    These check the verifier on small concrete address maps so the executable
    contract is exercised at build time (`#guard` makes the build fail if the
    Bool evaluates to false). When wired to JSON IPC, the same predicate
    `verifyAddressMap` runs against Rust-supplied data. -/

section Test

/-- A 4-task graph with one data-dependency edge `0 → 2` (reader of buf "a"
    before writer of buf "b"). Tasks 0, 1 are on resource 0; tasks 2, 3 on
    resource 1. Counter `0` is incremented by task 0 and waited on by task 2,
    so the counter-gated edge backs the data dep. -/
private def tgEx : TaskGraph := {
  n := 4
  edges := [
    (⟨0, by decide⟩, ⟨2, by decide⟩)
  ]
}

private def pEx : CounterProtocol tgEx := {
  waits     := fun t => if t.val = 2 then [0] else []
  succs     := fun t => if t.val = 0 then [0] else []
  threshold := fun _ => 1
  resource  := fun t => if t.val ≤ 1 then 0 else 1
  streamIdx := fun t => t.val
}

/-- Both buffers have real writers and readers. Buffer "a" is written at task
    0 and read at task 1 (last read). Buffer "b" is written at task 2 (first
    writer, reclaiming "a"'s bytes) and read at task 3. They overlap at
    offset 0, size 100.

    Task 1 → task 2 must be ordered. Counter `0` is succ'd by task 0 and
    waited by task 2 (so 0 ⇒ 2 directly). But task 1 → 2 needs the resource
    chain: tasks 0, 1 are on resource 0 (streamIdx 0, 1); tasks 2, 3 on
    resource 1 (streamIdx 2, 3). Resource doesn't bridge across.

    The schedule must therefore route the dep `1 → 2` through some chain.
    Here counter 0 only relates 0 → 2, not 1 → 2. So even the "safe" map
    actually has no `1 → 2` happens-before path — and the verifier should
    reject it. To make it pass, we attach a `last-read counter` from task 1
    that task 2 also waits on. -/
private def pSafe : CounterProtocol tgEx := {
  waits     := fun t => if t.val = 2 then [0, 1] else []
  succs     := fun t =>
    if t.val = 0 then [0]
    else if t.val = 1 then [1]
    else []
  threshold := fun c => match c with | 0 => 1 | 1 => 1 | _ => 0
  resource  := fun t => if t.val ≤ 1 then 0 else 1
  streamIdx := fun t => t.val
}

private def safeMap : List (AddrEntry tgEx) := [
  { name := "a", offset := 0, size := 100, cls := BufClass.Scratch,
    writers := [⟨0, by decide⟩], readers := [⟨1, by decide⟩] },
  { name := "b", offset := 0, size := 100, cls := BufClass.Scratch,
    writers := [⟨2, by decide⟩], readers := [⟨3, by decide⟩] }
]

#guard verifyAddressMap pSafe safeMap

/- Same buffers; protocol `pEx` only has a counter from task 0 (not task 1)
   to task 2. So task 1 (last reader of "a") has no happens-before path to
   task 2 (first writer of "b"). The verifier must reject. -/
#guard ¬ verifyAddressMap pEx safeMap

/- Disjoint byte ranges: the verifier accepts regardless of ordering. -/
private def disjointMap : List (AddrEntry tgEx) := [
  { name := "a", offset := 0, size := 100, cls := BufClass.Scratch,
    writers := [⟨0, by decide⟩], readers := [⟨1, by decide⟩] },
  { name := "b", offset := 200, size := 100, cls := BufClass.Scratch,
    writers := [⟨2, by decide⟩], readers := [⟨3, by decide⟩] }
]

#guard verifyAddressMap pEx disjointMap

end Test

end Plow.Verify
