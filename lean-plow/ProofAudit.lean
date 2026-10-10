/-
# `proof_audit` — axiom audit of the verifier's acceptance theorems.

Run as `lake exe proof_audit [proof-manifest.json]` after `lake build`. It imports the built
`Plow` environment and checks, for `proof-manifest.json`:

* the manifest's endpoint ids are exactly `Plow.CLI.Dispatch.endpointIds`;
* every listed theorem exists, is a theorem, and its transitive axiom closure is inside
  `allowed_axioms` (so no `sorryAx`, no project axiom, no `Lean.ofReduceBool`);
* a listed `checker` is a constant the theorem's statement mentions;
* the audit itself rejects a theorem built from `sorryAx` and one built from a fresh axiom.

Exit 0 iff all hold. The report lists every theorem's axiom set.
-/
import Lean
import Plow
import Plow.CLI.Dispatch

open Lean

structure Root where
  theorem_ : Name
  checker : Option Name
  /-- Expand to every theorem whose name has this string prefix (at least one). -/
  pfx : Option String := none

structure Endpoint where
  id : String
  scope : String
  roots : List Root

def parseName (s : String) : Name := s.toName

def isTheorem : ConstantInfo → Bool
  | .thmInfo _ => true
  | _ => false

def parseManifest (j : Json) : Except String (List Name × List Endpoint) := do
  let schema ← j.getObjValAs? Nat "schema"
  if schema != 1 then throw s!"unsupported manifest schema {schema}"
  let allowed ← j.getObjValAs? (List String) "allowed_axioms"
  let eps ← j.getObjValAs? (List Json) "endpoints"
  let eps ← eps.mapM fun e => do
    let id ← e.getObjValAs? String "id"
    let scope ← e.getObjValAs? String "scope"
    let roots ← e.getObjValAs? (List Json) "soundness"
    if roots.isEmpty then throw s!"endpoint {id}: no soundness theorem"
    let roots ← roots.mapM fun r => do
      if let .ok pfx := r.getObjValAs? String "theorem_prefix" then
        return { theorem_ := .anonymous, checker := none, pfx := some pfx : Root }
      let thm ← r.getObjValAs? String "theorem"
      let checker := (r.getObjValAs? String "checker").toOption
      pure { theorem_ := parseName thm, checker := checker.map parseName : Root }
    pure { id, scope, roots : Endpoint }
  pure (allowed.map parseName, eps)

def expand (env : Environment) (r : Root) : List Root :=
  match r.pfx with
  | none => [r]
  | some pfx =>
    let names := env.constants.fold (init := #[]) fun acc n info =>
      if isTheorem info && pfx.isPrefixOf n.toString then acc.push n else acc
    (names.qsort Name.lt).toList.map fun n => { theorem_ := n, checker := none }

def axiomsOf (env : Environment) (n : Name) : Array Name :=
  let ((), s) := ((CollectAxioms.collect n).run env).run {}
  s.axioms

/-- `none` = accepted; `some reason` = rejected. -/
def auditRoot (env : Environment) (allowed : List Name) (r : Root) : Option String × Array Name :=
  match env.find? r.theorem_ with
  | none => (some s!"{r.theorem_}: not found", #[])
  | some info =>
    if !isTheorem info then (some s!"{r.theorem_}: not a theorem", #[]) else
    let axs := axiomsOf env r.theorem_
    let bad := axs.filter (fun a => !allowed.contains a)
    if !bad.isEmpty then (some s!"{r.theorem_}: disallowed axioms {bad.toList}", axs) else
    match r.checker with
    | some c =>
      if (env.find? c).isNone then (some s!"{r.theorem_}: checker {c} not found", axs)
      else if !info.type.getUsedConstants.contains c then
        (some s!"{r.theorem_}: statement does not mention checker {c}", axs)
      else (none, axs)
    | none => (none, axs)

/-- The audit must reject `sorryAx` and a fresh axiom; otherwise its own acceptance means nothing. -/
def selfTest (env : Environment) (allowed : List Name) : Except String Unit := do
  let falseE := mkConst ``False
  let sorryThm : Declaration := .thmDecl {
    name := `ProofAudit.selfTest.usesSorry, levelParams := [], type := falseE,
    value := mkApp2 (mkConst ``sorryAx [levelZero]) falseE (mkConst ``Bool.false) }
  let freshAx : Declaration := .axiomDecl {
    name := `ProofAudit.selfTest.fresh, levelParams := [], type := falseE, isUnsafe := false }
  let axThm : Declaration := .thmDecl {
    name := `ProofAudit.selfTest.usesAxiom, levelParams := [], type := falseE,
    value := mkConst `ProofAudit.selfTest.fresh }
  let env ← match env.addDecl {} sorryThm with
    | .ok e => pure e | .error _ => throw "self-test: could not add sorry theorem"
  let env ← match env.addDecl {} freshAx with
    | .ok e => pure e | .error _ => throw "self-test: could not add axiom"
  let env ← match env.addDecl {} axThm with
    | .ok e => pure e | .error _ => throw "self-test: could not add axiom theorem"
  for n in [`ProofAudit.selfTest.usesSorry, `ProofAudit.selfTest.usesAxiom] do
    if (auditRoot env allowed { theorem_ := n, checker := none }).1.isNone then
      throw s!"self-test: audit accepted {n}"

def main (args : List String) : IO UInt32 := do
  let path := args.headD "proof-manifest.json"
  let manifest ← match Json.parse (← IO.FS.readFile path) with
    | .ok j => pure j
    | .error e => IO.eprintln s!"{path}: {e}"; return 2
  let (allowed, eps) ← match parseManifest manifest with
    | .ok m => pure m
    | .error e => IO.eprintln s!"{path}: {e}"; return 2
  initSearchPath (← findSysroot)
  let env ← importModules #[{ module := `Plow }, { module := `Plow.CLI.Dispatch }] {}
  let mut failures : Array String := #[]
  match selfTest env allowed with
  | .ok () => IO.println "self-test: sorryAx and fresh axioms rejected"
  | .error e => failures := failures.push e
  let ids := eps.map (·.id)
  for id in Plow.CLI.Dispatch.endpointIds do
    if !ids.contains id then failures := failures.push s!"endpoint {id}: dispatched but not in manifest"
  for id in ids do
    if !Plow.CLI.Dispatch.endpointIds.contains id then
      failures := failures.push s!"endpoint {id}: in manifest but not dispatched"
    if (ids.filter (· == id)).length != 1 then failures := failures.push s!"endpoint {id}: duplicated"
  let mut theorems := 0
  for e in eps do
    for r in e.roots do
      let rs := expand env r
      if rs.isEmpty then
        failures := failures.push s!"endpoint {e.id}: prefix {r.pfx.getD ""} matches no theorem"
      for r in rs do
        theorems := theorems + 1
        let (verdict, axs) := auditRoot env allowed r
        match verdict with
        | none => IO.println s!"ok   {e.id} [{e.scope}] {r.theorem_} axioms={axs.toList}"
        | some reason =>
          IO.println s!"FAIL {e.id} [{e.scope}] {reason}"
          failures := failures.push s!"endpoint {e.id}: {reason}"
  if failures.isEmpty then
    IO.println s!"proof audit: {eps.length} endpoints, {theorems} theorems, axioms within {allowed}"
    return 0
  for f in failures do IO.eprintln s!"proof audit: {f}"
  return 1
