/-
# Plow.Knobs.Ledger — checkpoint P, the performance certificate for a default flip.

A ledger entry is one arm's measurement of one rung: a metric, which direction is better, and its
samples (or, for older harnesses, only `n`, `median` and `mad`). A treatment names its control and
its repeated control; both ran in the same job on the same hardware.

The floor is `|median(ctrl) − median(ctrl2)| + 2·max(mad)`, exact over the rationals. A rung whose
floor cannot be computed — no repeated control, a missing MAD, fewer than three samples, arms from
different jobs or hardware — is `insufficient`, and an insufficient rung blocks the flip exactly as a
rejected one does.

`R0 → R1` is accepted iff every untouched rung keeps its digest, every touched rung measured better
than its control by more than the floor (or, with cited evidence, not worse beyond it), the tier-4
serving metrics are not worse beyond their floors when tier 4 is required, and numeric-changing
scopes carry passing correctness facts.

* `checkP_sound`: accepted ⇒ each of those conditions holds.
* `insufficient_blocks`: an insufficient rung means the flip is not accepted.
* `flip_non_regression`: under `DigestPerfInvariant` (performance is a function of the rung digest
  on fixed hardware), an accepted flip leaves every untouched rung's performance equal and every
  touched rung measured better.
* `carry_over`: a rung's verdict is a function of the ledger and the rung alone.
* `per_rung_argmin`: the per-rung choice is the knob-off value unless a candidate beats it beyond
  the floor; ties go to off.
-/
import Lean.Data.Json
import Plow.Knobs.Schema

namespace Plow.Knobs.Ledger

open Lean (Json JsonNumber)
open Plow.Knobs (field arrOf strOf natOf strsOf)

/-! ## Exact rationals -/

/-- `num / den`. Every value here is built from JSON decimals and `+ - * /2`, so `den > 0`. -/
structure Q where
  num : Int
  den : Nat
  deriving Repr, DecidableEq, Inhabited

namespace Q
def ofNat (n : Nat) : Q := ⟨n, 1⟩
def add (a b : Q) : Q := ⟨a.num * b.den + b.num * a.den, a.den * b.den⟩
def sub (a b : Q) : Q := ⟨a.num * b.den - b.num * a.den, a.den * b.den⟩
def mulNat (a : Q) (k : Nat) : Q := ⟨a.num * k, a.den⟩
def half (a : Q) : Q := ⟨a.num, a.den * 2⟩
def abs (a : Q) : Q := ⟨a.num.natAbs, a.den⟩
def lt (a b : Q) : Bool := decide (a.num * b.den < b.num * a.den)
def le (a b : Q) : Bool := decide (a.num * b.den ≤ b.num * a.den)
def max (a b : Q) : Q := if lt a b then b else a
def ofJson (n : JsonNumber) : Q := ⟨n.mantissa, 10 ^ n.exponent⟩

def render (a : Q) : String :=
  let scaled := (a.num * 1000) / a.den
  let sign := if scaled < 0 then "-" else ""
  let m := scaled.natAbs
  s!"{sign}{m / 1000}.{toString (m % 1000 + 1000) |>.drop 1}"
end Q

/-! ## Entries and statistics -/

inductive Better where
  | lower | higher
  deriving DecidableEq, Repr, Inhabited

structure Stats where
  n : Nat
  median : Q
  mad : Option Q
  deriving Repr, Inhabited

structure Entry where
  id : String
  job : String
  hardware : String
  rung : String
  metric : String
  better : Better
  samples : List Q
  stats : Option Stats
  controlOf : Option String
  repeatControlOf : Option String
  deriving Repr, Inhabited

def insertSorted (x : Q) : List Q → List Q
  | [] => [x]
  | y :: ys => if Q.le x y then x :: y :: ys else y :: insertSorted x ys

def sortQ : List Q → List Q
  | [] => []
  | x :: xs => insertSorted x (sortQ xs)

def medianOf (xs : List Q) : Option Q :=
  let s := sortQ xs
  let n := s.length
  if n = 0 then none
  else if n % 2 = 1 then s[n / 2]?
  else match s[n / 2 - 1]?, s[n / 2]? with
    | some a, some b => some (Q.half (Q.add a b))
    | _, _ => none

def madOf (xs : List Q) : Option Q := do
  let m ← medianOf xs
  medianOf (xs.map fun x => Q.abs (Q.sub x m))

/-- Samples win over recorded stats; a stats-only record contributes only what it states. -/
def statsOf (e : Entry) : Option Stats :=
  if e.samples.isEmpty then e.stats
  else (medianOf e.samples).map fun m => ⟨e.samples.length, m, madOf e.samples⟩

def minSamples : Nat := 3
def k : Nat := 2

/-- The noise floor of a treatment against its two controls, or why it cannot be computed. -/
def floorOf (c c2 t : Entry) : Except String (Q × Q × Q) := do
  unless c.job = c2.job ∧ c.job = t.job do
    throw s!"arms ran in different jobs ({c.job}, {c2.job}, {t.job})"
  unless c.hardware = c2.hardware ∧ c.hardware = t.hardware do
    throw "arms ran on different hardware"
  unless c.rung = t.rung ∧ c2.rung = t.rung ∧ c.metric = t.metric ∧ c2.metric = t.metric do
    throw "arms measured different rungs or metrics"
  let some sc := statsOf c | throw s!"{c.id}: no median"
  let some sc2 := statsOf c2 | throw s!"{c2.id}: no median"
  let some st := statsOf t | throw s!"{t.id}: no median"
  unless minSamples ≤ sc.n ∧ minSamples ≤ sc2.n ∧ minSamples ≤ st.n do
    throw s!"fewer than {minSamples} samples in an arm (n = {sc.n}, {sc2.n}, {st.n})"
  let (some m1, some m2, some m3) := (sc.mad, sc2.mad, st.mad)
    | throw "an arm records no MAD"
  let floor := Q.add (Q.abs (Q.sub sc.median sc2.median)) (Q.mulNat (Q.max m1 (Q.max m2 m3)) k)
  pure (floor, Q.half (Q.add sc.median sc2.median), st.median)

/-- Treatment beats the control mean by more than the floor, in the better direction. -/
def improves (b : Better) (floor ctrl treat : Q) : Bool :=
  match b with
  | .lower => Q.lt floor (Q.sub ctrl treat)
  | .higher => Q.lt floor (Q.sub treat ctrl)

/-- Treatment is not worse than the control mean beyond the floor. -/
def notWorse (b : Better) (floor ctrl treat : Q) : Bool :=
  match b with
  | .lower => Q.le (Q.sub treat ctrl) floor
  | .higher => Q.le (Q.sub ctrl treat) floor

/-! ## Requests and verdicts -/

structure Touched where
  rung : String
  treat : String
  /-- Evidence that a neutral result is acceptable for this rung; empty: it must improve. -/
  neutral : List String
  deriving Repr, Inhabited

structure Untouched where
  rung : String
  base : String
  variant : String
  deriving Repr, Inhabited

structure Fact where
  kind : String
  pass : Bool
  evidence : String
  deriving Repr, Inhabited

structure Req where
  ledger : List Entry
  touched : List Touched
  untouched : List Untouched
  tier4 : Bool
  serving : List String
  numeric : Bool
  facts : List Fact

inductive Verdict where
  | accept (note : String)
  | reject (why : String)
  | insufficient (why : String)
  deriving Repr, Inhabited

def Verdict.isAccept : Verdict → Bool
  | .accept _ => true
  | _ => false

def lookup (l : List Entry) (id : String) : Option Entry := l.find? (·.id = id)

/-- Resolve a treatment's two controls and floor. -/
def armsOf (l : List Entry) (treat : String) : Except String (Entry × Q × Q × Q) := do
  let some t := lookup l treat | throw s!"no ledger entry {treat}"
  let some cid := t.controlOf | throw s!"{treat} names no control"
  let some c := lookup l cid | throw s!"no ledger entry {cid}"
  let some c2id := t.repeatControlOf | throw s!"{treat} names no repeated control"
  let some c2 := lookup l c2id | throw s!"no ledger entry {c2id}"
  let (floor, ctrl, tr) ← floorOf c c2 t
  pure (t, floor, ctrl, tr)

def rungVerdict (l : List Entry) (x : Touched) : Verdict :=
  match armsOf l x.treat with
  | .error why => .insufficient s!"{x.rung}: {why}"
  | .ok (t, floor, ctrl, tr) =>
    if t.rung ≠ x.rung then .insufficient s!"{x.rung}: {x.treat} measured {t.rung}"
    else if improves t.better floor ctrl tr then
      .accept s!"{x.rung}: {t.metric} {ctrl.render} -> {tr.render} (floor {floor.render})"
    else if !x.neutral.isEmpty && notWorse t.better floor ctrl tr then
      .accept s!"{x.rung}: neutral within floor {floor.render}, cited {x.neutral}"
    else .reject s!"{x.rung}: {t.metric} {ctrl.render} -> {tr.render} is not better beyond the floor {floor.render}"

def servingVerdict (l : List Entry) (treat : String) : Verdict :=
  match armsOf l treat with
  | .error why => .insufficient s!"serving {treat}: {why}"
  | .ok (t, floor, ctrl, tr) =>
    if notWorse t.better floor ctrl tr then
      .accept s!"serving {t.metric}: {ctrl.render} -> {tr.render} (floor {floor.render})"
    else .reject s!"serving {t.metric}: {ctrl.render} -> {tr.render} is worse beyond the floor {floor.render}"

def untouchedVerdict (u : Untouched) : Verdict :=
  if u.base = u.variant then .accept s!"{u.rung}: digest unchanged"
  else .reject s!"{u.rung}: digest changed although the scope leaves it untouched"

def factsVerdict (r : Req) : Verdict :=
  if !r.numeric then .accept "no numeric-changing field in scope"
  else if r.facts.isEmpty then .insufficient "the scope changes numerics and no correctness fact is recorded"
  else match r.facts.find? (!·.pass) with
    | some f => .reject s!"correctness fact {f.kind} failed: {f.evidence}"
    | none => .accept s!"correctness facts pass: {r.facts.map (·.kind)}"

def tier4Verdict (r : Req) : Verdict :=
  if !r.tier4 then .accept "tier 4 not required"
  else if r.serving.isEmpty then .insufficient "tier 4 is required and no serving measurement is named"
  else .accept "tier 4 measured"

def verdicts (r : Req) : List Verdict :=
  r.untouched.map untouchedVerdict ++ r.touched.map (rungVerdict r.ledger) ++
  r.serving.map (servingVerdict r.ledger) ++ [tier4Verdict r, factsVerdict r]

def checkP (r : Req) : Bool := (verdicts r).all Verdict.isAccept

/-! ## Theorems -/

theorem mem_verdicts_of_touched (r : Req) (x : Touched) (hx : x ∈ r.touched) :
    rungVerdict r.ledger x ∈ verdicts r := by
  simp only [verdicts, List.mem_append, List.mem_map]
  exact Or.inl (Or.inl (Or.inr ⟨x, hx, rfl⟩))

theorem checkP_sound (r : Req) (h : checkP r = true) :
    (∀ u ∈ r.untouched, u.base = u.variant) ∧
    (∀ x ∈ r.touched, (rungVerdict r.ledger x).isAccept = true) ∧
    (∀ s ∈ r.serving, (servingVerdict r.ledger s).isAccept = true) ∧
    (r.tier4 = true → r.serving ≠ []) ∧
    (r.numeric = true → r.facts ≠ [] ∧ ∀ f ∈ r.facts, f.pass = true) := by
  have hall := List.all_eq_true.mp h
  refine ⟨fun u hu => ?_, fun x hx => hall _ (mem_verdicts_of_touched r x hx), fun s hs => ?_, ?_, ?_⟩
  · have := hall (untouchedVerdict u) (by
      simp only [verdicts, List.mem_append, List.mem_map]
      exact Or.inl (Or.inl (Or.inl ⟨u, hu, rfl⟩)))
    unfold untouchedVerdict at this
    split at this
    · assumption
    · simp [Verdict.isAccept] at this
  · exact hall _ (by
      simp only [verdicts, List.mem_append, List.mem_map]
      exact Or.inl (Or.inr ⟨s, hs, rfl⟩))
  · intro ht
    have := hall (tier4Verdict r) (by simp [verdicts])
    unfold tier4Verdict at this
    simp only [ht, Bool.not_true, Bool.false_eq_true, if_false] at this
    split at this
    · simp [Verdict.isAccept] at this
    · rename_i hs; simpa using hs
  · intro hn
    have := hall (factsVerdict r) (by simp [verdicts])
    unfold factsVerdict at this
    simp only [hn, Bool.not_true, Bool.false_eq_true, if_false] at this
    split at this
    · simp [Verdict.isAccept] at this
    · rename_i hne
      split at this
      · simp [Verdict.isAccept] at this
      · rename_i hfind
        refine ⟨by simpa using hne, fun f hf => ?_⟩
        rw [List.find?_eq_none] at hfind
        simpa using hfind f hf

/-- An insufficient rung blocks the flip. -/
theorem insufficient_blocks (r : Req) (x : Touched) (hx : x ∈ r.touched) (why : String)
    (hi : rungVerdict r.ledger x = .insufficient why) : checkP r = false := by
  cases h : checkP r with
  | false => rfl
  | true =>
    have := (checkP_sound r h).2.1 x hx
    rw [hi] at this
    simp [Verdict.isAccept] at this

/-- An accepted touched rung without a neutral exception measured an improvement beyond its
    floor, computed from its own controls. -/
theorem accept_is_improvement (l : List Entry) (x : Touched) (note : String)
    (hn : x.neutral = []) (ha : rungVerdict l x = .accept note) :
    ∃ t floor ctrl tr, armsOf l x.treat = .ok (t, floor, ctrl, tr) ∧
      improves t.better floor ctrl tr = true := by
  unfold rungVerdict at ha
  split at ha
  · cases ha
  · rename_i t floor ctrl tr heq
    refine ⟨t, floor, ctrl, tr, heq, ?_⟩
    split at ha
    · cases ha
    · split at ha
      · assumption
      · simp [hn] at ha

/-- Performance on fixed hardware is a function of the rung digest. -/
def DigestPerfInvariant (perf : String → Q) (measured : String → String → Q) : Prop :=
  ∀ rung d, measured rung d = perf d

/-- Under the invariant, an accepted flip leaves every untouched rung's performance where it was,
    and every touched rung carries an accepted measurement. -/
theorem flip_non_regression (r : Req) (perf : String → Q) (measured : String → String → Q)
    (hinv : DigestPerfInvariant perf measured) (h : checkP r = true) :
    (∀ u ∈ r.untouched, measured u.rung u.base = measured u.rung u.variant) ∧
    (∀ x ∈ r.touched, (rungVerdict r.ledger x).isAccept = true) := by
  obtain ⟨hu, ht, _⟩ := checkP_sound r h
  exact ⟨fun u hmem => by rw [hinv, hinv, hu u hmem], ht⟩

/-- A rung's verdict depends only on the ledger and the rung: a measurement carries over to every
    recipe whose rung it names. -/
theorem carry_over (r₁ r₂ : Req) (x : Touched) (hl : r₁.ledger = r₂.ledger) :
    rungVerdict r₁.ledger x = rungVerdict r₂.ledger x := by rw [hl]

structure Arm where
  value : String
  beats : Bool
  median : Q
  deriving Repr, Inhabited

def pick (best a : Arm) : Arm := if Q.lt a.median best.median then a else best

/-- The per-rung choice: the best candidate that beats knob-off beyond its floor, else knob-off. -/
def argmin (off : Arm) (cands : List Arm) : Arm :=
  match cands.filter (·.beats) with
  | [] => off
  | c :: cs => cs.foldl pick c

theorem foldl_best_mem : ∀ (c : Arm) (cs : List Arm), cs.foldl pick c ∈ c :: cs
  | _, [] => by simp
  | c, a :: as => by
    simp only [List.foldl_cons]
    rcases List.mem_cons.mp (foldl_best_mem (pick c a) as) with h | h
    · rw [h]; unfold pick; split <;> simp
    · exact List.mem_cons_of_mem _ (List.mem_cons_of_mem _ h)

theorem per_rung_argmin (off : Arm) (cands : List Arm) :
    ((cands.filter (·.beats)) = [] → argmin off cands = off) ∧
    (argmin off cands = off ∨ (argmin off cands ∈ cands ∧ (argmin off cands).beats = true)) := by
  constructor
  · intro h; simp [argmin, h]
  · unfold argmin
    split
    · exact Or.inl rfl
    · rename_i c cs hf
      right
      have hmem : c :: cs ⊆ cands.filter (·.beats) := by rw [hf]; exact fun _ h => h
      have hsel := hmem (foldl_best_mem c cs)
      simpa using hsel

/-! ## JSON -/

def parseQ (ctx : String) (j : Json) : Except String Q :=
  match j with
  | .num n => .ok (Q.ofJson n)
  | _ => .error s!"{ctx}: expected a number"

def optStr (j : Json) (k : String) : Option String :=
  match j.getObjVal? k with
  | .ok (.str s) => some s
  | _ => none

def parseEntry (j : Json) : Except String Entry := do
  let id ← strOf "id" (← field j "id")
  let hw ← field j "hardware"
  let rung ← field j "rung"
  let better ← match ← strOf "better" (← field j "better") with
    | "lower" => pure Better.lower
    | "higher" => pure Better.higher
    | b => throw s!"{id}: unknown direction '{b}'"
  let samples ← match j.getObjVal? "samples" with
    | .ok s => (← arrOf "samples" s).mapM (parseQ s!"{id} sample")
    | .error _ => pure []
  let stats ← match j.getObjVal? "stats" with
    | .error _ | .ok .null => pure none
    | .ok s => pure (some {
        n := ← natOf "stats.n" (← field s "n"),
        median := ← parseQ "stats.median" (← field s "median"),
        mad := ← match s.getObjVal? "mad" with
          | .ok (.num n) => pure (some (Q.ofJson n))
          | _ => pure none })
  pure { id, job := ← strOf "job" (← field j "job"),
         hardware := hw.compress, rung := ← strOf "rung.digest" (← field rung "digest"),
         metric := ← strOf "metric" (← field j "metric"), better, samples, stats,
         controlOf := optStr j "control_of", repeatControlOf := optStr j "repeat_control_of" }

def parseReq (j : Json) : Except String Req := do
  let ledger ← (← arrOf "ledger" (← field j "ledger")).mapM parseEntry
  let touched ← (← arrOf "touched" (← field j "touched")).mapM fun x => do
    pure { rung := ← strOf "rung" (← field x "rung"), treat := ← strOf "treat" (← field x "treat"),
           neutral := ← match x.getObjVal? "neutral_evidence" with
             | .ok e => strsOf "neutral_evidence" e
             | .error _ => pure [] }
  let untouched ← (← arrOf "untouched" (← field j "untouched")).mapM fun u => do
    pure { rung := ← strOf "rung" (← field u "rung"), base := ← strOf "base" (← field u "base"),
           variant := ← strOf "variant" (← field u "variant") }
  let facts ← match j.getObjVal? "facts" with
    | .error _ => pure []
    | .ok fs => (← arrOf "facts" fs).mapM fun f => do
      pure { kind := ← strOf "kind" (← field f "kind"),
             pass := (f.getObjValAs? Bool "pass").toOption.getD false,
             evidence := (optStr f "evidence").getD "" }
  pure { ledger, touched, untouched,
         tier4 := (j.getObjValAs? Bool "tier4").toOption.getD false,
         serving := ← match j.getObjVal? "serving" with
           | .ok s => strsOf "serving" s
           | .error _ => pure [],
         numeric := (j.getObjValAs? Bool "numeric").toOption.getD false,
         facts }

def render : Verdict → String
  | .accept n => s!"accept: {n}"
  | .reject w => s!"reject: {w}"
  | .insufficient w => s!"insufficient_evidence: {w}"

def runP (payload : Json) : Except String String := do
  let r ← parseReq payload
  let vs := verdicts r
  let lines := String.intercalate " | " (vs.map render)
  if checkP r then return s!"flip accepted: {lines}"
  else if vs.any (fun v => match v with | .reject _ => true | _ => false) then
    throw s!"flip rejected: {lines}"
  else throw s!"insufficient_evidence: {lines}"

end Plow.Knobs.Ledger
