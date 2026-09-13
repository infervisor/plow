/-
# Plow.Knobs.Consistency — checkpoint K.

`checkK` is the proven core: a well-formed registry, in-domain sources, the resolved config, and
every constraint evaluated on it. `runK` adds the instance checks around it that carry no
universal claim: unknown sources, registry consistency over the declared targets (recipes
satisfiable, qualified defaults reachable, no knob whose "on" contradicts the constraints,
evidence cited), agreement with the emitter's own record, and a dead-knob report.
-/
import Lean.Data.Json
import Plow.Knobs.Resolve

namespace Plow.Knobs

open Lean (Json)

def firstViolation (c : Config) (t : Target) : List Constraint → Option String
  | [] => none
  | x :: xs => if eval c t x.formula then firstViolation c t xs else some x.id

theorem firstViolation_none (c : Config) (t : Target) : ∀ cs : List Constraint,
    firstViolation c t cs = none → ∀ x ∈ cs, eval c t x.formula = true
  | [], _, _, hx => absurd hx (List.not_mem_nil _)
  | y :: ys, h, x, hx => by
    by_cases hy : eval c t y.formula = true
    · simp only [firstViolation, hy, if_true] at h
      rcases List.mem_cons.mp hx with rfl | hx
      · exact hy
      · exact firstViolation_none c t ys h x hx
    · simp [firstViolation, hy] at h

def checkK (reg : Registry) (src : Sources) (t : Target) : Except String Config :=
  if ordered [] reg.specs = false then
    .error "registry: a knob id repeats, or a production default reads a knob declared after it"
  else if defaultsInDomain reg.specs = false then
    .error "registry: a default value is outside its knob's domain"
  else if sourcesInDomain reg.specs src = false then
    .error "a cli/env value is outside its knob's domain"
  else
    match firstViolation (resolve reg.specs src t) t reg.constraints with
    | some id => .error s!"constraint {id} is violated by the resolved config"
    | none => .ok (resolve reg.specs src t)

/-- Checkpoint K soundness: an accepted config is the resolution of these sources, satisfies every
    registered constraint, and gives every registered knob a value in its domain. -/
theorem checkK_sound (reg : Registry) (src : Sources) (t : Target) (c : Config)
    (h : checkK reg src t = .ok c) :
    c = resolve reg.specs src t ∧
    (∀ x ∈ reg.constraints, eval c t x.formula = true) ∧
    (∀ k ∈ reg.specs, ∃ v, find c k.id = some v ∧ admits k.domain v = true) := by
  unfold checkK at h
  cases hwf : ordered [] reg.specs <;> simp only [hwf, if_true, if_false] at h
  · cases h
  cases hdef : defaultsInDomain reg.specs <;> simp only [hdef, if_true, if_false] at h
  · cases h
  cases hsrc : sourcesInDomain reg.specs src <;> simp only [hsrc, if_true, if_false] at h
  · cases h
  cases hv : firstViolation (resolve reg.specs src t) t reg.constraints with
  | some id => simp [hv] at h
  | none =>
    simp only [hv, reduceCtorEq] at h
    cases h
    exact ⟨rfl, firstViolation_none _ t _ hv,
      fun k hk => resolve_total reg.specs src t k hwf hdef hsrc hk⟩

/-! ## Instance checks (no universal claim) -/

def statusName : Status → String
  | .qualified _ => "qualified" | .optIn => "opt_in" | .candidate _ => "candidate"
  | .parked _ _ => "parked"
  | .diagnostic => "diagnostic" | .removed => "removed"

def firstBadSource (reg : Registry) (src : Sources) : Option String :=
  (reg.specs.find? fun k =>
    !(optAdmits k.domain (srcGet src k.id).cli && optAdmits k.domain (srcGet src k.id).env)).map
    fun k => k.id

def unknownSources (reg : Registry) (src : Sources) : List String :=
  src.filterMap fun (id, _) => if reg.specs.any (·.id == id) then none else some id

def candidates (reg : Registry) (x : String) : List Val :=
  let consts := reg.constraints.foldl (fun acc c => acc ++ constsFor x c.formula) []
  let base : List Val := match reg.specs.find? (·.id == x) with
    | some { domain := .bool, .. } => [.bool false, .bool true]
    | some { domain := .nat lo hi, .. } =>
      let ns := consts.foldl (fun acc v => match v with
        | .nat n => acc ++ [n, n + 1] ++ (if n > 0 then [n - 1] else [])
        | _ => acc) [lo, hi]
      (ns.filter fun n => lo ≤ n && n ≤ hi).map .nat
    | some { domain := .enum vs, .. } => vs.map .str
    | _ => .str "" :: consts.filter fun | .str _ => true | _ => false
  (Val.unset :: base).eraseDups

/-- The knobs transitively connected to `start` through shared constraints. -/
def component (reg : Registry) (start : String) : List String := Id.run do
  let mut seen := [start]
  for _ in [0:reg.constraints.length + 1] do
    for c in reg.constraints do
      let vs := vars c.formula
      if vs.any (memB · seen) then
        for v in vs do
          if !memB v seen then seen := seen ++ [v]
  return seen

def assignments : List (String × List Val) → List Config
  | [] => [[]]
  | (x, vs) :: rest => (assignments rest).flatMap fun a => vs.map fun v => (x, v) :: a

def enumCap : Nat := 200000

/-- Some config agreeing with the target's recipe outside `start`'s component, with `start = v`,
    satisfies every constraint: `some true`/`some false`; `none` when the space is too large. -/
def satisfiableWith (reg : Registry) (t : Target) (recipe : Sources) (start : String) (v : Val) :
    Option Bool :=
  let base := resolve reg.specs recipe t
  let others := (component reg start).filter (· != start)
  let space := others.map fun x => (x, candidates reg x)
  let size := space.foldl (fun n (_, vs) => n * vs.length) 1
  if size > enumCap then none
  else some <| (assignments space).any fun a =>
    reg.constraints.all fun c => eval ((start, v) :: a ++ base) t c.formula

def boolKnobsInConstraints (reg : Registry) : List KnobSpec :=
  reg.specs.filter fun k =>
    (match k.domain with | .bool => true | _ => false) &&
    reg.constraints.any fun c => memB k.id (vars c.formula)

structure Consistency where
  errors : List String
  dead : List String
  skipped : List String

def registryConsistency (reg : Registry) : Consistency := Id.run do
  let mut errors : List String := []
  let mut dead : List String := []
  let mut skipped : List String := []
  for (t, recipe) in reg.targets do
    match checkK reg recipe t with
    | .error e => errors := errors ++ [s!"target {t.name}: qualified recipe rejected: {e}"]
    | .ok _ => pure ()
  for k in reg.specs do
    match k.status with
    | .qualified [] => errors := errors ++ [s!"{k.id}: qualified without evidence"]
    | .candidate [] => errors := errors ++ [s!"{k.id}: candidate without evidence"]
    | .parked _ [] => errors := errors ++ [s!"{k.id}: parked without evidence"]
    | _ => pure ()
    match k.status, k.dflt with
    | .qualified _, .production (_ :: _) o =>
      let reachable := reg.targets.any fun (t, recipe) =>
        get (resolve reg.specs (recipe.filter (·.1 != k.id)) t) k.id != o
      if !reachable then
        errors := errors ++ [s!"{k.id}: qualified production default fires on no declared target"]
    | _, _ => pure ()
  for k in boolKnobsInConstraints reg do
    let verdicts := reg.targets.map fun (t, recipe) => satisfiableWith reg t recipe k.id (.bool true)
    if verdicts.any (· == some true) then pure ()
    else if verdicts.any (· == none) then skipped := skipped ++ [k.id]
    else match k.status with
      | .qualified _ | .optIn | .candidate _ =>
        errors := errors ++ [s!"{k.id}={Val.render (.bool true)} contradicts the constraints on \
          every declared target ({statusName k.status})"]
      | _ => dead := dead ++ [k.id]
  return { errors, dead, skipped }

def parseRecorded (j : Json) : Except String (List (String × Val)) := do
  match j.getObjVal? "recorded" with
  | .error _ => pure []
  | .ok r => (← arrOf "recorded" r).mapM fun e => do
      pure (← strOf "recorded id" (← field e "id"), ← parseVal (← field e "value"))

def verdict (reg : Registry) (src : Sources) (t : Target) : String :=
  match checkK reg src t with
  | .ok _ => "ok"
  | .error e =>
    if e.startsWith "constraint " then (e.drop 11).takeWhile (· != ' ') else "wf"

/-- The payload handler. `cases` switches to batch mode for differential testing: every case is
    decided by `checkK` alone and the notes carry one verdict per case. -/
def runK (payload : Json) : Except String String := do
  let reg ← parseRegistry payload
  if let .ok cs := payload.getObjVal? "cases" then
    let verdicts ← (← arrOf "cases" cs).mapM fun c => do
      pure (verdict reg (← parseSources (← field c "sources")) (← parseTarget (← field c "target")))
    return s!"verdicts={String.intercalate ";" verdicts}"
  let t ← parseTarget (← field payload "target")
  let src ← parseSources (← field payload "sources")
  let unknown := unknownSources reg src
  if !unknown.isEmpty then
    throw s!"sources name knobs the registry does not declare: {unknown}"
  if sourcesInDomain reg.specs src = false then
    throw s!"source value outside its domain: {firstBadSource reg src}"
  let cfg ← checkK reg src t
  let mismatches := (← parseRecorded payload).filterMap fun (id, v) =>
    let r := get cfg id
    if r = v then none else some s!"{id}: resolved {r.render}, emitter recorded {v.render}"
  if !mismatches.isEmpty then
    throw s!"the registry resolves differently from the emitter's record — the registry's defaults \
      are stale: {mismatches}"
  let cons := registryConsistency reg
  if !cons.errors.isEmpty then
    throw s!"registry inconsistent: {cons.errors}"
  let notes := s!"{reg.specs.length} knobs resolved; {reg.constraints.length} constraints hold on \
    {t.arch}/tp{t.tp}/{t.model}; {reg.targets.length} declared targets consistent"
  let notes := if cons.dead.isEmpty then notes else notes ++ s!"; dead knobs: {cons.dead}"
  let notes := if cons.skipped.isEmpty then notes else notes ++ s!"; satisfiability not enumerated: {cons.skipped}"
  return notes

end Plow.Knobs
