/-
# Plow.Knobs.Schema — the knob registry, a target, knob sources, and their JSON decoders.

Mirrors `plow_asset::knob` on the Rust side. A knob id is layer-qualified (`emit.fp8`,
`rt.mla_pf_v2`, `env.PLOW_BLOCK`, `def.PLOW_GLM_OFOLD`), so one registry spans every layer.
-/
import Lean.Data.Json

namespace Plow.Knobs

open Lean (Json)

inductive Val where
  | unset
  | bool (b : Bool)
  | nat (n : Nat)
  | str (s : String)
  deriving DecidableEq, Repr, Inhabited

def Val.render : Val → String
  | .unset => "unset"
  | .bool b => toString b
  | .nat n => toString n
  | .str s => s!"\"{s}\""

inductive Domain where
  | bool
  | nat (lo hi : Nat)
  | enum (vs : List String)
  | list (vs : List String)
  | str
  deriving Repr, Inhabited

inductive Cmp where
  | eq | ne | lt | le | gt | ge
  deriving DecidableEq, Repr, Inhabited

inductive TAtom where
  | arch (s : String)
  | tp (n : Nat)
  | nCu (n : Nat)
  | model (s : String)
  | cap (s : String)
  deriving Repr, Inhabited

inductive Formula where
  | tt
  | atom (k : String) (c : Cmp) (v : Val)
  | tgt (a : TAtom)
  | not (f : Formula)
  | and (a b : Formula)
  | or (a b : Formula)
  | implies (a b : Formula)
  deriving Repr, Inhabited

inductive Layer where
  | emit | runtime | objectDefine | rawEnv
  deriving DecidableEq, Repr, Inhabited

inductive Status where
  | qualified (evidence : List String)
  | optIn
  | candidate (evidence : List String)
  | parked (reason : String) (evidence : List String)
  | diagnostic
  | removed
  deriving Repr, Inhabited

structure DefaultCase where
  when : Formula
  value : Val
  deriving Repr, Inhabited

/-- `production cases otherwise`: the first case whose `when` holds, else `otherwise`. -/
inductive Dflt where
  | static (v : Val)
  | production (cases : List DefaultCase) (otherwise : Val)
  deriving Repr, Inhabited

structure KnobSpec where
  id : String
  layer : Layer
  domain : Domain
  dflt : Dflt
  status : Status
  deriving Repr, Inhabited

structure Constraint where
  id : String
  formula : Formula
  deriving Repr, Inhabited

structure Target where
  name : String
  arch : String
  tp : Nat
  nCu : Nat
  model : String
  caps : List String
  deriving Repr, Inhabited

structure Source where
  cli : Option Val := none
  env : Option Val := none
  deriving Repr, Inhabited

abbrev Config := List (String × Val)
abbrev Sources := List (String × Source)

/-- `targets` are the declared targets, each with its qualified recipe. -/
structure Registry where
  specs : List KnobSpec
  constraints : List Constraint
  targets : List (Target × Sources)

/-! ## JSON decoders -/

def field (j : Json) (k : String) : Except String Json :=
  match j.getObjVal? k with
  | .ok v => .ok v
  | .error _ => .error s!"missing field '{k}' in {j.compress.take 200}"

def arrOf (ctx : String) (j : Json) : Except String (List Json) :=
  match j with
  | .arr a => .ok a.toList
  | _ => .error s!"{ctx}: expected an array"

def strOf (ctx : String) (j : Json) : Except String String :=
  match j with
  | .str s => .ok s
  | _ => .error s!"{ctx}: expected a string"

def natOf (ctx : String) (j : Json) : Except String Nat :=
  match j.getNat? with
  | .ok n => .ok n
  | .error _ => .error s!"{ctx}: expected a natural number"

def strsOf (ctx : String) (j : Json) : Except String (List String) := do
  (← arrOf ctx j).mapM (strOf ctx)

def parseVal (j : Json) : Except String Val :=
  match j with
  | .null => .ok .unset
  | .bool b => .ok (.bool b)
  | .str s => .ok (.str s)
  | .num _ => (natOf "value" j).map .nat
  | _ => .error s!"value {j.compress}: expected null, bool, nat or string"

def parseCmp : String → Except String Cmp
  | "eq" => .ok .eq | "ne" => .ok .ne | "lt" => .ok .lt
  | "le" => .ok .le | "gt" => .ok .gt | "ge" => .ok .ge
  | s => .error s!"unknown comparison '{s}'"

/-- `["true"]`, `["atom", id, cmp, value]`, `["arch", s]`, `["tp", n]`, `["n_cu", n]`,
    `["model", s]`, `["cap", s]`, `["not", f]`, `["and", [f..]]`, `["or", [f..]]`,
    `["implies", a, b]`. -/
partial def parseFormula (j : Json) : Except String Formula := do
  let xs ← arrOf "formula" j
  let arg (i : Nat) : Except String Json :=
    match xs[i]? with
    | some x => .ok x
    | none => .error s!"formula {j.compress.take 120}: missing argument {i}"
  let tag ← strOf "formula tag" (← arg 0)
  match tag with
  | "true" => pure .tt
  | "atom" => pure (.atom (← strOf "atom id" (← arg 1)) (← parseCmp (← strOf "cmp" (← arg 2)))
                        (← parseVal (← arg 3)))
  | "arch" => pure (.tgt (.arch (← strOf "arch" (← arg 1))))
  | "tp" => pure (.tgt (.tp (← natOf "tp" (← arg 1))))
  | "n_cu" => pure (.tgt (.nCu (← natOf "n_cu" (← arg 1))))
  | "model" => pure (.tgt (.model (← strOf "model" (← arg 1))))
  | "cap" => pure (.tgt (.cap (← strOf "cap" (← arg 1))))
  | "not" => pure (.not (← parseFormula (← arg 1)))
  | "and" => do
      let fs ← (← arrOf "and" (← arg 1)).mapM parseFormula
      pure (fs.foldr .and .tt)
  | "or" => do
      let fs ← (← arrOf "or" (← arg 1)).mapM parseFormula
      pure (fs.foldr .or (.not .tt))
  | "implies" => pure (.implies (← parseFormula (← arg 1)) (← parseFormula (← arg 2)))
  | t => throw s!"unknown formula tag '{t}'"

def parseDomain (j : Json) : Except String Domain := do
  match ← strOf "domain.kind" (← field j "kind") with
  | "bool" => pure .bool
  | "nat" => pure (.nat (← natOf "min" (← field j "min")) (← natOf "max" (← field j "max")))
  | "enum" => pure (.enum (← strsOf "values" (← field j "values")))
  | "list" => pure (.list (← strsOf "values" (← field j "values")))
  | "str" => pure .str
  | k => throw s!"unknown domain kind '{k}'"

def parseLayer : String → Except String Layer
  | "emit" => .ok .emit | "runtime" => .ok .runtime
  | "object_define" => .ok .objectDefine | "raw_env" => .ok .rawEnv
  | s => .error s!"unknown layer '{s}'"

def parseStatus (j : Json) : Except String Status := do
  match ← strOf "status.kind" (← field j "kind") with
  | "qualified" => pure (.qualified (← strsOf "evidence" (← field j "evidence")))
  | "opt_in" => pure .optIn
  | "candidate" => pure (.candidate (← strsOf "evidence" (← field j "evidence")))
  | "parked" => pure (.parked (← strOf "reason" (← field j "reason"))
                               (← strsOf "evidence" (← field j "evidence")))
  | "diagnostic" => pure .diagnostic
  | "removed" => pure .removed
  | k => throw s!"unknown status kind '{k}'"

def parseDflt (j : Json) : Except String Dflt := do
  match j.getObjVal? "static" with
  | .ok v => pure (.static (← parseVal v))
  | .error _ =>
    let cases ← (← arrOf "production" (← field j "production")).mapM fun c => do
      pure { when := ← parseFormula (← field c "when"), value := ← parseVal (← field c "value") }
    pure (.production cases (← parseVal (← field j "otherwise")))

def parseSpec (j : Json) : Except String KnobSpec := do
  let id ← strOf "id" (← field j "id")
  let ctx (e : String) := s!"knob {id}: {e}"
  let layer ← (parseLayer (← strOf "layer" (← field j "layer"))).mapError ctx
  let domain ← (parseDomain (← field j "domain")).mapError ctx
  let dflt ← (parseDflt (← field j "default")).mapError ctx
  let status ← (parseStatus (← field j "status")).mapError ctx
  pure { id, layer, domain, dflt, status }

def parseConstraint (j : Json) : Except String Constraint := do
  let id ← strOf "constraint id" (← field j "id")
  pure { id, formula := ← (parseFormula (← field j "formula")).mapError (s!"constraint {id}: " ++ ·) }

def optVal (j : Json) (k : String) : Except String (Option Val) :=
  match j.getObjVal? k with
  | .ok v => (parseVal v).map some
  | .error _ => .ok none

def parseSources (j : Json) : Except String Sources := do
  (← arrOf "sources" j).mapM fun s => do
    pure (← strOf "source id" (← field s "id"), { cli := ← optVal s "cli", env := ← optVal s "env" })

def parseTarget (j : Json) : Except String Target := do
  pure { name := (← (j.getObjValAs? String "name").toOption.getD "" |> pure),
         arch := ← strOf "arch" (← field j "arch"),
         tp := ← natOf "tp" (← field j "tp"),
         nCu := ← natOf "n_cu" (← field j "n_cu"),
         model := ← strOf "model" (← field j "model"),
         caps := ← strsOf "caps" (← field j "caps") }

def parseRegistry (j : Json) : Except String Registry := do
  let specs ← (← arrOf "registry" (← field j "registry")).mapM parseSpec
  let constraints ← (← arrOf "constraints" (← field j "constraints")).mapM parseConstraint
  let targets ← match j.getObjVal? "targets" with
    | .error _ => pure []
    | .ok ts => (← arrOf "targets" ts).mapM fun t => do
        let recipe ← match t.getObjVal? "recipe" with
          | .ok r => parseSources r
          | .error _ => pure []
        pure (← parseTarget t, recipe)
  pure { specs, constraints, targets }

end Plow.Knobs
